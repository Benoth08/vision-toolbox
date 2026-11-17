// Lecture de model_card.json : tout ce que le programme doit savoir sur le modele, ecrit par la toolbox.
using System.Text.Json; // lecture du JSON
using System.Text.Json.Serialization; // attribut JsonPropertyName : nom du champ dans le fichier JSON

namespace Vtb; // espace de noms commun a tout le programme

// Chaque propriete ci-dessous est remplie avec le champ JSON de meme nom (JsonPropertyName) ; { get; set; } = lisible et modifiable.
public sealed class InputSpec // description de l'entree attendue par le modele (sealed : pas de classe derivee)
{
    [JsonPropertyName("name")] public string Name { get; set; } = "input"; // nom de l'entree du modele ONNX
    [JsonPropertyName("height")] public int Height { get; set; } // hauteur de l'image d'entree en pixels
    [JsonPropertyName("width")] public int Width { get; set; } // largeur en pixels
    [JsonPropertyName("channels")] public int Channels { get; set; } // nombre de canaux : 1 (gris) ou 3 (couleur)
    [JsonPropertyName("color")] public string Color { get; set; } = "RGB";          // RGB, GRAY ou BGR
    [JsonPropertyName("scale")] public string Scale { get; set; } = "";             // "none ..." : pas de division
    [JsonPropertyName("mean")] public float[] Mean { get; set; } = []; // moyenne par canal a soustraire (normalisation)
    [JsonPropertyName("std")] public float[] Std { get; set; } = []; // ecart-type par canal par lequel diviser
    [JsonPropertyName("resize")] public string Resize { get; set; } = "stretch";    // stretch, letterbox, letterbox_top_left, none
    // letterbox : image reduite sans deformation puis completee par des bandes ; ici la valeur de ces bandes (int? : peut manquer)
    [JsonPropertyName("letterbox_pad_value")] public int? LetterboxPadValue { get; set; }
    [JsonPropertyName("read_color")] public string? ReadColor { get; set; }        // "gray" ou "rgb" : lecture du fichier
}

public sealed class ModelCard // la fiche du modele (model_card.json)
{
    [JsonPropertyName("task")] public string Task { get; set; } = ""; // tache : classification, anomaly, segmentation ou detection
    [JsonPropertyName("model_name")] public string ModelName { get; set; } = ""; // nom du modele, ex. "ultralytics:yolo11n"
    [JsonPropertyName("input")] public InputSpec Input { get; set; } = new(); // description de l'entree (classe ci-dessus)
    // liste des operations de pretraitement a appliquer dans l'ordre (nom + parametres)
    [JsonPropertyName("preprocess")] public List<Dictionary<string, JsonElement>> Preprocess { get; set; } = [];
    [JsonPropertyName("class_names")] public List<string> ClassNames { get; set; } = []; // noms des classes, dans l'ordre des sorties
    [JsonPropertyName("thresholds")] public Dictionary<string, double> Thresholds { get; set; } = []; // seuils de decision (conf, image...)
    [JsonPropertyName("files")] public Dictionary<string, string> Files { get; set; } = []; // fichiers associes (ex. "onnx" -> model.onnx)

    // lit la fiche depuis un fichier ; renvoie l'objet ModelCard
    public static ModelCard Load(string path) =>
        // convertit le JSON en ModelCard ; ?? throw : erreur si le resultat est vide (null)
        JsonSerializer.Deserialize<ModelCard>(File.ReadAllText(path)) ?? throw new InvalidDataException($"carte illisible : {path}");

    /// <summary>Decodeur de detection : prefixe de model_name ("ultralytics", "yolox", "fasterrcnn").</summary>
    public string Decoder => Task == "detection" ? ModelName.Split(':')[0] : ""; // texte avant ":" pour la detection, "" sinon
}
