// Comparaison aux reponses de reference ecrites par tools/make_golden.py (le predicteur Python sans PyTorch).
// Deux niveaux : le tenseur d'entree (pretraitement identique ?) puis la reponse (post-traitement identique ?).
using System.Text.Json; // lecture du fichier golden.json

namespace Vtb; // espace de noms commun a tout le programme

public static class Golden // classe "outil" (static : pas besoin de creer d'objet) pour la verification golden
{
    // compare chaque image de reference ; renvoie 0 si tout est identique a Python, 1 sinon (code de sortie du programme)
    public static int Check(Predictor predictor, string goldenPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(goldenPath)); // lit et analyse le JSON ; libere a la fin de la methode
        string root = Path.GetDirectoryName(Path.GetFullPath(goldenPath))!; // dossier du fichier : les chemins d'images y sont relatifs
        int failures = 0; // nombre d'images en ecart
        foreach (var item in doc.RootElement.GetProperty("images").EnumerateArray()) // pour chaque image de reference
        {
            string file = Path.Combine(root, item.GetProperty("file").GetString()!); // chemin complet de l'image
            var image = predictor.Read(file); // lit l'image comme le fait Python
            var tensor = predictor.Prepare(image, out _); // pretraitement : image -> tenseur d'entree (out _ : on ignore les infos annexes)
            double sum = 0, expectedSum = item.GetProperty("tensor_sum").GetDouble(); // somme calculee ici, et somme attendue (Python)
            foreach (var v in tensor) sum += v; // additionne toutes les valeurs du tenseur
            double tolerance = 1e-3 * Math.Max(1.0, Math.Abs(expectedSum)) + 1e-2; // petit ecart permis (arrondis des calculs en float)
            bool inputOk = Math.Abs(sum - expectedSum) <= tolerance; // vrai si le pretraitement donne le meme tenseur qu'en Python
            var prediction = predictor.Predict(image); // reponse complete du modele en C#
            bool outputOk = CompareResult(predictor.Card.Task, prediction, item.GetProperty("result")); // compare a la reponse Python
            // affiche une ligne par image : nom, verdict sur l'entree (avec les sommes en cas d'ecart), verdict sur la sortie
            Console.WriteLine($"{Path.GetFileName(file)}\tentree {(inputOk ? "OK" : $"ECART {sum:F3} / {expectedSum:F3}")}\tsortie {(outputOk ? "OK" : "ECART")}");
            if (!inputOk || !outputOk) failures++; // une image de plus en ecart si l'un des deux niveaux differe
        }
        // bilan final
        Console.WriteLine(failures == 0 ? "golden : tout est identique a Python" : $"golden : {failures} image(s) en ecart");
        return failures == 0 ? 0 : 1; // code de sortie : 0 = succes, 1 = ecart
    }

    // compare la reponse C# a la reponse Python selon la tache ; renvoie vrai si elles concordent
    static bool CompareResult(string task, Prediction p, JsonElement expected) => task switch
    {
        // classification : meme classe, et probabilite egale a 0,001 pres
        "classification" => p.Label == expected.GetProperty("label").GetString() && Math.Abs(p.Probability!.Value - expected.GetProperty("probability").GetDouble()) < 1e-3,
        // anomalie : meme score (ecart relatif 0,1 %) et meme decision anomalie oui / non
        "anomaly" => Math.Abs(p.Score!.Value - expected.GetProperty("score").GetDouble()) <= 1e-3 * Math.Max(1.0, Math.Abs(p.Score.Value)) && p.Flagged == expected.GetProperty("anomaly").GetBoolean(),
        // segmentation : meme nombre de pixels de defaut, a 2 pixels pres
        "segmentation" => Math.Abs(p.DefectPixels!.Value - expected.GetProperty("defect_pixels").GetInt32()) <= 2,
        "detection" => p.Objects.Count == expected.GetProperty("count").GetInt32(), // detection : meme nombre d'objets trouves
        _ => false, // tache inconnue : on ne sait pas comparer, donc ecart
    };
}
