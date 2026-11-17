// Portage C# de vtb/serve/predictor.py : meme entree, meme post-traitement, memes reponses.
// ONNX : format de fichier standard pour un modele entraine ; ONNX Runtime l'execute sans Python ni PyTorch.
using Microsoft.ML.OnnxRuntime; // ONNX Runtime : charge et execute le modele .onnx
using Microsoft.ML.OnnxRuntime.Tensors; // tenseurs d'ONNX Runtime (tableaux de nombres a plusieurs dimensions)
using OpenCvSharp; // OpenCvSharp : OpenCV pour C# (lecture d'images, redimensionnement, couleurs)

namespace Vtb; // espace de noms commun a tout le programme

// un objet detecte : classe, score de confiance, coins haut-gauche (X1, Y1) et bas-droite (X2, Y2) de sa boite en pixels
public sealed record Detection(string Label, float Score, float X1, float Y1, float X2, float Y2);

public sealed class Prediction // reponse du modele pour une image ; seuls les champs de la tache sont remplis
{
    public string Task { get; init; } = ""; // tache (init : fixe a la creation, ne change plus ensuite)
    public string? Label { get; init; }                 // classification
    public double? Probability { get; init; } // classification : probabilite de la classe choisie (double? : peut etre vide)
    public double? Score { get; init; }                 // anomalie
    public bool? Flagged { get; init; }                 // anomalie, segmentation
    public int? DefectPixels { get; init; }             // segmentation
    public List<Detection> Objects { get; init; } = []; // detection
}

public sealed class Predictor : IDisposable // charge le modele et fait les predictions ; IDisposable : libere la memoire native a la fin
{
    readonly InferenceSession _session; // session ONNX Runtime : le modele charge, pret a calculer
    readonly string _inputName; // nom de l'entree du modele (a donner a chaque appel)
    public ModelCard Card { get; } // fiche du modele (pretraitement, classes, seuils)

    public Predictor(string modelDir) // constructeur : charge la fiche et le modele depuis le dossier exporte
    {
        Card = ModelCard.Load(Path.Combine(modelDir, "model_card.json")); // lit la fiche du modele
        string onnx = Path.Combine(modelDir, Card.Files.GetValueOrDefault("onnx", "model.onnx")); // fichier ONNX (model.onnx par defaut)
        _session = new InferenceSession(onnx); // charge le modele dans ONNX Runtime
        _inputName = _session.InputMetadata.Keys.First(); // nom de la premiere (et seule) entree du modele
    }

    public void Dispose() => _session.Dispose(); // libere la session (memoire native) quand on a fini

    /// <summary>Lit une image comme vtb.io.images.read_image : gris ou RGB, 8 ou 16 bits conserves.</summary>
    public Mat Read(string path) // renvoie l'image lue (Mat = image OpenCV)
    {
        // lecture en gris si la fiche le demande (read_color, sinon couleur d'entree GRAY)
        bool gray = Card.Input.ReadColor is not null ? Card.Input.ReadColor == "gray" : Card.Input.Color == "GRAY";
        var mode = (gray ? ImreadModes.Grayscale : ImreadModes.Color) | ImreadModes.AnyDepth; // AnyDepth : garde le 16 bits
        var image = Cv2.ImRead(path, mode); // lit le fichier
        if (image.Empty()) throw new IOException($"image illisible : {path}"); // image vide = fichier absent ou illisible : erreur claire
        return gray ? image : image.CvtColor(ColorConversionCodes.BGR2RGB); // OpenCV lit en BGR : on passe en RGB comme Python
    }

    sealed class Meta // informations gardees entre le pretraitement et le post-traitement
    {
        // tailles d'origine et apres pretraitement, decalages du letterbox (en pixels)
        public int OriginalH, OriginalW, ProcessedH, ProcessedW, PadX, PadY;
        public double Ratio = 1.0; // facteur d'echelle du letterbox (pour remettre les boites a la bonne taille)
    }

    // Prepare : renvoie le tenseur d'entree du modele ; metaOut : infos gardees pour le post-traitement
    /// <summary>Pipeline de la carte -> mise a l'echelle d'entree -> couleur -> valeurs -> mean / std -> NCHW float32.</summary>
    public DenseTensor<float> Prepare(Mat image, out object metaOut)
    {
        var spec = Card.Input; // description de l'entree attendue
        var processed = Preprocessing.Apply(image, Card.Preprocess); // applique les operations de pretraitement de la fiche
        // retient les tailles d'origine et apres pretraitement
        var meta = new Meta { OriginalH = image.Rows, OriginalW = image.Cols, ProcessedH = processed.Rows, ProcessedW = processed.Cols };
        var x = processed; // image de travail
        // modele couleur mais image grise : on recopie le gris sur 3 canaux
        if (spec.Channels == 3 && x.Channels() == 1) x = x.CvtColor(ColorConversionCodes.GRAY2RGB);
        int pad = spec.LetterboxPadValue ?? 114; // valeur des bandes du letterbox (114 = gris, convention YOLO, si absente)
        if (spec.Resize == "letterbox") // letterbox : reduire sans deformer, puis centrer sur un fond uni
        {
            // Comme vtb letterbox_params : ratio, taille arrondie, bordure centree.
            meta.Ratio = Math.Min((double)spec.Height / x.Rows, (double)spec.Width / x.Cols); // plus grand facteur qui fait tenir l'image
            // nouvelles dimensions, arrondies comme round() de Python (au pair le plus proche en cas d'egalite)
            int newW = (int)Math.Round(x.Cols * meta.Ratio, MidpointRounding.ToEven), newH = (int)Math.Round(x.Rows * meta.Ratio, MidpointRounding.ToEven);
            meta.PadX = (spec.Width - newW) / 2; // largeur de la bande de gauche
            meta.PadY = (spec.Height - newH) / 2; // hauteur de la bande du haut
            var resized = new Mat(); // image redimensionnee
            Cv2.Resize(x, resized, new Size(newW, newH), 0, 0, InterpolationFlags.Linear); // redimensionne (interpolation lineaire)
            var canvas = new Mat(spec.Height, spec.Width, x.Type(), Scalar.All(pad)); // fond uni a la taille d'entree du modele
            resized.CopyTo(new Mat(canvas, new Rect(meta.PadX, meta.PadY, newW, newH))); // colle l'image au centre du fond
            x = canvas; // l'image de travail devient le fond complete
        }
        else if (spec.Resize == "letterbox_top_left") // variante YOLOX : image collee en haut a gauche
        {
            // YOLOX : int() tronque, image collee en haut a gauche.
            meta.Ratio = Math.Min((double)spec.Height / x.Rows, (double)spec.Width / x.Cols); // facteur qui fait tenir l'image
            // nouvelles dimensions, tronquees comme int() en Python
            int newW = (int)(x.Cols * meta.Ratio), newH = (int)(x.Rows * meta.Ratio);
            var resized = new Mat(); // image redimensionnee
            Cv2.Resize(x, resized, new Size(newW, newH), 0, 0, InterpolationFlags.Linear); // redimensionne
            var canvas = new Mat(spec.Height, spec.Width, x.Type(), Scalar.All(pad)); // fond uni
            resized.CopyTo(new Mat(canvas, new Rect(0, 0, newW, newH))); // colle l'image dans le coin haut-gauche
            x = canvas; // l'image de travail devient le fond complete
        }
        if (spec.Color == "BGR") x = x.CvtColor(ColorConversionCodes.RGB2BGR); // certains modeles (YOLOX) attendent l'ordre BGR

        double divisor = spec.Scale.StartsWith("none") ? 1.0 : Preprocessing.Peak(x); // diviseur : 255 (ou 65535), ou 1 si "none"
        var floats = new Mat(); // image en nombres decimaux
        x.ConvertTo(floats, MatType.CV_32FC(x.Channels()), 1.0 / divisor); // convertit en float32 et ramene les valeurs dans [0, 1]
        if (spec.Resize == "stretch" && (floats.Rows != spec.Height || floats.Cols != spec.Width)) // etirement si la taille differe
        {
            // RF-DETR : taille fixe sans garder les proportions, sur les valeurs deja en flottant (comme Python)
            var resized = new Mat(); // image etiree
            Cv2.Resize(floats, resized, new Size(spec.Width, spec.Height), 0, 0, InterpolationFlags.Linear); // etire a la taille exacte
            floats = resized; // remplace l'image de travail
        }
        int c = floats.Channels(), h = floats.Rows, w = floats.Cols; // canaux, hauteur, largeur
        if (!floats.IsContinuous()) floats = floats.Clone(); // GetArray exige des pixels contigus en memoire : copie si besoin
        floats.Reshape(1).GetArray(out float[] hwc); // 1 canal de largeur W x C : ordre HWC, comme numpy
        // NCHW : ordre attendu par le modele = (lot N=1, canaux C, hauteur H, largeur W), alors qu'OpenCV range en HWC
        var tensor = new DenseTensor<float>(new[] { 1, c, h, w });
        for (int k = 0; k < c; k++) // pour chaque canal
        {
            // normalisation : moyenne et ecart-type du canal (0 et 1 si la fiche n'en donne pas)
            float mean = spec.Mean.Length > 0 ? spec.Mean[k] : 0f, std = spec.Std.Length > 0 ? spec.Std[k] : 1f;
            for (int i = 0; i < h; i++) // pour chaque ligne
                for (int j = 0; j < w; j++) // pour chaque colonne
                    tensor[0, k, i, j] = (hwc[(i * w + j) * c + k] - mean) / std; // range le pixel en NCHW, normalise
        }
        metaOut = meta; // rend les infos a l'appelant
        return tensor; // renvoie le tenseur pret pour le modele
    }

    public Prediction Predict(Mat image) // prediction complete pour une image ; renvoie la reponse
    {
        var tensor = Prepare(image, out var metaObj); // pretraitement
        var meta = (Meta)metaObj; // retrouve le type Meta des infos
        using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) }); // execute le modele ONNX
        if (Card.Task == "detection") return DetectionResult(results, meta); // Faster R-CNN : labels en int64
        var outputs = results.ToDictionary(r => r.Name, r => r.AsTensor<float>()); // sorties du modele : nom -> tenseur float
        return Card.Task switch // decode selon la tache
        {
            "classification" => Classification(outputs), // classe la plus probable
            "anomaly" => Anomaly(outputs), // score d'anomalie
            "segmentation" => Segmentation(outputs), // surface de defaut
            _ => throw new NotSupportedException(Card.Task), // tache inconnue : erreur
        };
    }

    Prediction Classification(Dictionary<string, Tensor<float>> outputs) // decode une classification ; renvoie classe et probabilite
    {
        var logits = outputs.Values.First().ToArray(); // logits : scores bruts du modele, un par classe (avant softmax)
        // softmax (transforme les scores en probabilites qui somment a 1) : on retire le max pour eviter les debordements
        double max = logits.Max(), sum = logits.Sum(v => Math.Exp(v - max));
        int best = Array.IndexOf(logits, logits.Max()); // indice de la classe au plus grand score
        // reponse : nom de la classe et sa probabilite softmax
        return new Prediction { Task = "classification", Label = Card.ClassNames[best], Probability = Math.Exp(logits[best] - max) / sum };
    }

    Prediction Anomaly(Dictionary<string, Tensor<float>> outputs) // decode une detection d'anomalie ; renvoie score et decision
    {
        double score = outputs["score"][0]; // score d'anomalie de l'image (plus haut = plus anormal)
        // reponse : anomalie si le score depasse le seuil choisi
        return new Prediction { Task = "anomaly", Score = score, Flagged = score > Card.Thresholds["image"] };
    }

    Prediction Segmentation(Dictionary<string, Tensor<float>> outputs) // decode une segmentation ; renvoie le nombre de pixels de defaut
    {
        var probs = outputs["probabilities"]; // probabilites par pixel : (1, classes, hauteur, largeur)
        int k = probs.Dimensions[1], h = probs.Dimensions[2], w = probs.Dimensions[3], area = 0; // classes, taille, compteur de pixels
        double threshold = Card.Thresholds["defect_probability"]; // seuil de probabilite pour dire "defaut" (cas a 2 classes)
        for (int i = 0; i < h; i++) // pour chaque ligne
            for (int j = 0; j < w; j++) // pour chaque colonne
            {
                if (k == 2) { if (probs[0, 1, i, j] > threshold) area++; continue; } // 2 classes : defaut si proba > seuil
                int best = 0; // plus de 2 classes : on cherche la classe la plus probable du pixel
                for (int c = 1; c < k; c++) if (probs[0, c, i, j] > probs[0, best, i, j]) best = c; // garde la plus probable
                if (best > 0) area++; // classe 0 = fond ; toute autre classe = defaut
            }
        // reponse : surface de defaut, et decision "defectueux" si elle atteint la surface minimale
        return new Prediction { Task = "segmentation", DefectPixels = area, Flagged = area >= Card.Thresholds.GetValueOrDefault("min_defect_area", 0) };
    }

    // decode une detection d'objets selon la famille de modele ; renvoie la liste des boites
    Prediction DetectionResult(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, Meta meta)
    {
        // seuil de confiance ; seuil d'IoU (recouvrement = intersection / union de deux boites) pour la NMS
        double conf = Card.Thresholds["conf"], iou = Card.Thresholds.GetValueOrDefault("iou_nms", 0.6);
        var rows = new List<float[]>(); // x1, y1, x2, y2, score, classe (repere de l'image pretraitee)
        var first = results.First().AsTensor<float>(); // premiere sortie du modele
        if (Card.Decoder == "ultralytics") // YOLO11 (Ultralytics)
        {
            int attrs = first.Dimensions[1], anchors = first.Dimensions[2]; // (1, 4 + classes, ancres) ; ancre = une boite candidate
            for (int a = 0; a < anchors; a++) // pour chaque boite candidate
            {
                int best = 0; float score = float.MinValue; // meilleure classe et son score
                // meilleure classe : on parcourt les scores des classes, places apres les 4 coordonnees
                for (int c = 4; c < attrs; c++) if (first[0, c, a] > score) { score = first[0, c, a]; best = c - 4; }
                if (score <= conf) continue; // trop peu sur : on ignore (strictement superieur, comme Python)
                float cx = first[0, 0, a], cy = first[0, 1, a], bw = first[0, 2, a], bh = first[0, 3, a]; // centre, largeur, hauteur
                rows.Add([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, best]); // converti en coins (x1, y1, x2, y2)
            }
            rows = Nms.PerClass(rows, iou); // NMS : supprime les boites en double sur un meme objet
            foreach (var r in rows) // pour chaque boite gardee : on annule le letterbox
            {
                // x1 et x2 : on retire la bande de gauche puis on divise par le facteur d'echelle
                r[0] = (float)((r[0] - meta.PadX) / meta.Ratio); r[2] = (float)((r[2] - meta.PadX) / meta.Ratio);
                r[1] = (float)((r[1] - meta.PadY) / meta.Ratio); r[3] = (float)((r[3] - meta.PadY) / meta.Ratio); // y : idem
            }
        }
        else if (Card.Decoder == "yolox") // YOLOX : sorties brutes par case de grille, a decoder
        {
            int size = Card.Input.Height, attrs = first.Dimensions[2], a = 0; // taille d'entree (carree), attributs, index de prediction
            foreach (int stride in new[] { 8, 16, 32 }) // pour chaque niveau de grille (stride = taille d'une case en pixels)
            {
                int n = size / stride; // nombre de cases par cote
                for (int gy = 0; gy < n; gy++) // pour chaque ligne de cases
                    for (int gx = 0; gx < n; gx++, a++) // pour chaque colonne (a avance d'une prediction a chaque case)
                    {
                        int best = 0; float prob = float.MinValue; // meilleure classe et sa probabilite
                        // meilleure classe : on parcourt les probabilites apres les 4 coordonnees et le score "objet"
                        for (int c = 5; c < attrs; c++) if (first[0, a, c] > prob) { prob = first[0, a, c]; best = c - 5; }
                        float score = first[0, a, 4] * prob; // score = "il y a un objet" x "c'est cette classe"
                        if (score < conf) continue; // trop peu sur : on ignore
                        // centre de la boite : (decalage dans la case + numero de case) x stride = position en pixels
                        float cx = (first[0, a, 0] + gx) * stride, cy = (first[0, a, 1] + gy) * stride;
                        // taille de la boite : exp(sortie) x stride
                        float bw = MathF.Exp(first[0, a, 2]) * stride, bh = MathF.Exp(first[0, a, 3]) * stride;
                        rows.Add([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, score, best]); // converti en coins
                    }
            }
            rows = Nms.PerClass(rows, iou); // supprime les doublons
            // annule la mise a l'echelle (pas de bande a retirer : l'image est collee en haut a gauche)
            foreach (var r in rows) for (int i = 0; i < 4; i++) r[i] = (float)(r[i] / meta.Ratio);
        }
        else if (Card.Decoder == "rfdetr") // RF-DETR : une prediction par "requete", pas de doublons a supprimer
        {
            // DETR : sigmoide, 300 meilleurs couples (requete, classe), pas de NMS ; boites cxcywh normalisees
            var named = results.ToDictionary(r => r.Name); // sorties par nom
            var dets = named["dets"].AsTensor<float>(); // boites : (1, requetes, 4) en centre / taille, entre 0 et 1
            var logits = named["labels"].AsTensor<float>(); // scores bruts : (1, requetes, classes)
            int queries = logits.Dimensions[1], classes = logits.Dimensions[2], numClasses = Card.ClassNames.Count; // dimensions utiles
            var scored = new List<(float Score, int Query, int Label)>(); // tous les couples (score, requete, classe)
            for (int q = 0; q < queries; q++) // pour chaque requete
                for (int k = 0; k < classes; k++) // pour chaque classe
                    scored.Add(((float)(1.0 / (1.0 + Math.Exp(-logits[0, q, k]))), q, k)); // sigmoide : score brut -> probabilite 0-1
            // les 300 meilleurs couples, du plus sur au moins sur (a egalite : ordre d'origine, comme numpy)
            foreach (var (score, q, k) in scored.OrderByDescending(t => t.Score).ThenBy(t => t.Query * classes + t.Label).Take(300))
            {
                if (score < conf || k >= numClasses) continue; // trop peu sur, ou classe hors liste : ignore
                float cx = dets[0, q, 0], cy = dets[0, q, 1], bw = dets[0, q, 2], bh = dets[0, q, 3]; // centre et taille normalises
                // coins en pixels de l'image pretraitee (on multiplie par sa largeur / hauteur)
                rows.Add([(cx - bw / 2) * meta.ProcessedW, (cy - bh / 2) * meta.ProcessedH, (cx + bw / 2) * meta.ProcessedW, (cy + bh / 2) * meta.ProcessedH, score, k]);
            }
        }
        else if (Card.Decoder == "fasterrcnn") // Faster R-CNN : le modele exporte donne deja les boites finales
        {
            var named = results.ToDictionary(r => r.Name); // sorties par nom
            var boxes = named["boxes"].AsTensor<float>(); // boites (x1, y1, x2, y2)
            var scores = named["scores"].AsTensor<float>(); // scores de confiance
            var labels = named["labels"].AsTensor<long>(); // classes, en entiers 64 bits
            for (int i = 0; i < scores.Dimensions[0]; i++) // pour chaque boite
                // gardee seulement si son score atteint le seuil de confiance
                if (scores[i] >= conf) rows.Add([boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i], labels[i]]);
        }
        else throw new NotSupportedException($"decodeur inconnu : {Card.ModelName}"); // famille de modele inconnue : erreur

        float sx = (float)meta.OriginalW / meta.ProcessedW, sy = (float)meta.OriginalH / meta.ProcessedH; // echelle pretraitee -> origine
        var objects = rows.Select(r => new Detection( // transforme chaque ligne en objet Detection
            Card.ClassNames[(int)r[5]], r[4], // nom de la classe et score
            Math.Clamp(r[0] * sx, 0, meta.OriginalW), Math.Clamp(r[1] * sy, 0, meta.OriginalH), // coin haut-gauche, garde dans l'image
            Math.Clamp(r[2] * sx, 0, meta.OriginalW), Math.Clamp(r[3] * sy, 0, meta.OriginalH))).ToList(); // coin bas-droite, idem
        return new Prediction { Task = "detection", Objects = objects }; // reponse : la liste des objets
    }
}

public static class Nms // NMS (non-maximum suppression) : garde la meilleure boite et supprime celles qui la recouvrent trop
{
    static float Iou(float[] a, float[] b) // IoU de deux boites : surface commune / surface totale (0 = disjointes, 1 = identiques)
    {
        // surface de l'intersection (0 si les boites ne se touchent pas)
        float inter = Math.Max(0, Math.Min(a[2], b[2]) - Math.Max(a[0], b[0])) * Math.Max(0, Math.Min(a[3], b[3]) - Math.Max(a[1], b[1]));
        float union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter; // surface de l'union
        return union > 0 ? inter / union : 0; // rapport, 0 si les boites sont vides
    }

    /// <summary>NMS classe par classe, scores decroissants (comme predictor.nms_per_class).</summary>
    public static List<float[]> PerClass(List<float[]> rows, double iou) // renvoie les boites gardees, triees par score
    {
        var kept = new List<float[]>(); // boites gardees
        foreach (var group in rows.GroupBy(r => r[5])) // pour chaque classe (colonne 5)
        {
            var order = group.OrderByDescending(r => r[4]).ToList(); // boites de la classe, du meilleur score au moins bon
            while (order.Count > 0) // tant qu'il reste des boites a examiner
            {
                var best = order[0]; // la meilleure restante
                kept.Add(best); // est gardee
                order = order.Skip(1).Where(r => Iou(best, r) <= iou).ToList(); // on retire celles qui la recouvrent trop (doublons)
            }
        }
        return kept.OrderByDescending(r => r[4]).ToList(); // toutes classes confondues, triees par score decroissant
    }
}
