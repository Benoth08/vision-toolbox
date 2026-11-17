// Ligne de commande :
//   dotnet run -- --model <dossier exporte> --image <fichier ou dossier>
//   dotnet run -- --model <dossier exporte> --golden <golden.json>    (compare aux reponses Python)
// Programme de production : execute le modele ONNX exporte par la toolbox, sans Python ni PyTorch.
using System.Text.Json; // ecriture des resultats en JSON
using Vtb; // nos classes : Predictor, Golden, Prediction...

var options = new Dictionary<string, string>(); // options lues sur la ligne de commande : nom -> valeur
// lit les arguments deux par deux ("--model" puis sa valeur) ; TrimStart retire les tirets du nom
for (int i = 0; i + 1 < args.Length; i += 2) options[args[i].TrimStart('-')] = args[i + 1];
if (!options.ContainsKey("model")) // si l'option --model manque
{
    Console.Error.WriteLine("usage : VtbInference --model <dossier> (--image <fichier|dossier> | --golden <golden.json>)"); // aide
    return 2; // code de sortie 2 = mauvaise utilisation
}

using var predictor = new Predictor(options["model"]); // charge le modele et sa fiche ; libere a la fin du programme
// reglages JSON : sur une seule ligne, sans les champs vides (null)
var json = new JsonSerializerOptions { WriteIndented = false, DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull };

if (options.TryGetValue("golden", out var goldenPath)) // si --golden est donne
    return Golden.Check(predictor, goldenPath); // mode verification : compare aux reponses Python et s'arrete

string source = options["image"]; // fichier ou dossier d'images a traiter
var files = Directory.Exists(source) // si c'est un dossier...
    // ...toutes ses images (extensions reconnues), triees par nom
    ? Directory.EnumerateFiles(source).Where(f => new[] { ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff" }.Contains(Path.GetExtension(f).ToLowerInvariant())).Order()
    : new[] { source }.AsEnumerable(); // ...sinon une liste d'un seul fichier
foreach (var file in files) // pour chaque image
{
    var watch = System.Diagnostics.Stopwatch.StartNew(); // chronometre demarre
    var result = predictor.Predict(predictor.Read(file)); // lit l'image puis fait la prediction
    // affiche : nom du fichier, duree en millisecondes, resultat en JSON
    Console.WriteLine($"{Path.GetFileName(file)}\t{watch.Elapsed.TotalMilliseconds:F1} ms\t{JsonSerializer.Serialize(result, json)}");
}
return 0; // code de sortie 0 = succes
