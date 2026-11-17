// Portage C# des operations de vtb.preprocess les plus utilisees, a l'identique de la version Python.
// Une operation absente leve NotSupportedException : la porter depuis src/vtb/preprocess/ops (chaque
// docstring Python donne l'equivalent OpenCvSharp), puis verifier avec les fichiers de reference (--golden).
using System.Text.Json; // lecture des parametres des operations (JsonElement)
using OpenCvSharp; // OpenCvSharp : OpenCV pour C# (flous, contraste, redimensionnement)

namespace Vtb; // espace de noms commun a tout le programme

public static class Preprocessing // operations de pretraitement (static : pas besoin de creer d'objet)
{
    // applique les operations de la fiche, dans l'ordre ; renvoie l'image transformee
    public static Mat Apply(Mat image, IReadOnlyList<Dictionary<string, JsonElement>> steps)
    {
        var current = image; // image de travail
        foreach (var step in steps) // pour chaque operation de la liste
        {
            string name = step["name"].GetString()!; // nom de l'operation (ex. "clahe")
            current = name switch // choisit la fonction selon le nom
            {
                "resize" => Resize(current, step), // redimensionnement
                "crop" => Crop(current, step), // decoupe d'un rectangle
                "to_gray" => current.Channels() == 1 ? current : current.CvtColor(ColorConversionCodes.RGB2GRAY), // passage en gris
                "gaussian_blur" => GaussianBlur(current, Num(step, "sigma", 1.0)), // flou gaussien (sigma = force du flou)
                "median_blur" => MedianBlur(current, (int)Num(step, "ksize", 3)), // flou median (enleve les points isoles)
                "clahe" => Clahe(current, Num(step, "clip_limit", 2.0), (int)Num(step, "tile_grid", 8)), // contraste local (CLAHE)
                "gamma" => Gamma(current, Num(step, "gamma", 1.0)), // correction gamma (eclaircit ou assombrit les tons moyens)
                // etirement du contraste entre deux percentiles
                "percentile_stretch" => PercentileStretch(current, Num(step, "low", 1.0), Num(step, "high", 99.0)),
                // correction d'eclairage : divise l'image par une estimation lissee du fond
                "shading_division" => ShadingDivision(current, Num(step, "sigma", 50.0), (int)Num(step, "downscale", 4), Bool(step, "keep_mean", true), Str(step, "method", "gaussian")),
                // operation inconnue : erreur qui dit ou trouver de quoi la porter
                _ => throw new NotSupportedException($"operation '{name}' non portee en C# : voir src/vtb/preprocess/ops (docstring : equivalent OpenCvSharp)"),
            };
        }
        return current; // renvoie l'image apres toutes les operations
    }

    // --- aides ------------------------------------------------------------------------------------

    public static double Peak(Mat m) => m.Depth() == MatType.CV_16U ? 65535.0 : 255.0; // valeur maximale : 65535 en 16 bits, 255 sinon

    // lit un parametre numerique de l'operation ; renvoie la valeur par defaut s'il manque
    static double Num(Dictionary<string, JsonElement> p, string key, double fallback) =>
        p.TryGetValue(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : fallback; // present et nombre ? sinon defaut

    // lit un parametre vrai / faux ; renvoie la valeur par defaut s'il manque
    static bool Bool(Dictionary<string, JsonElement> p, string key, bool fallback) =>
        // present et booleen (true ou false) ? sinon defaut
        p.TryGetValue(key, out var v) && (v.ValueKind == JsonValueKind.True || v.ValueKind == JsonValueKind.False) ? v.GetBoolean() : fallback;

    // lit un parametre texte ; renvoie la valeur par defaut s'il manque
    static string Str(Dictionary<string, JsonElement> p, string key, string fallback) =>
        p.TryGetValue(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : fallback; // present et texte ? sinon defaut

    /// <summary>Flottant [0, 1] -> type d'origine, comme vtb.preprocess.dtypes.from_float (ecretage puis arrondi + 0,5).</summary>
    static Mat FromFloat(Mat values, Mat like) // renvoie l'image reconvertie dans le type de "like" (8 ou 16 bits)
    {
        double peak = Peak(like); // 255 ou 65535
        var clipped = new Mat(); // valeurs ecretees
        Cv2.Min(values, 1.0, clipped); // ecretage : tout ce qui depasse 1 devient 1
        Cv2.Max(clipped, 0.0, clipped); // et tout ce qui est sous 0 devient 0
        var dst = new Mat(); // image resultat
        // Python : (v * peak + 0,5) tronque = arrondi ; ConvertTo arrondit deja (pas de +0,5 ici, sinon decalage d'un niveau).
        clipped.ConvertTo(dst, like.Type(), peak, 0); // multiplie par peak et convertit dans le type d'origine
        return dst; // renvoie l'image reconvertie
    }

    static Mat AsFloat(Mat m) // convertit une image 8 / 16 bits en flottants dans [0, 1] ; renvoie la nouvelle image
    {
        var dst = new Mat(); // image resultat
        m.ConvertTo(dst, MatType.CV_32FC(m.Channels()), 1.0 / Peak(m)); // float32, meme nombre de canaux, divise par la valeur max
        return dst; // renvoie l'image en flottants
    }

    // --- operations -----------------------------------------------------------------------------

    static Mat Resize(Mat src, Dictionary<string, JsonElement> p) // redimensionne ; renvoie la nouvelle image
    {
        int h, w; // hauteur et largeur voulues
        var size = p["size"]; // parametre size : un nombre (carre) ou [hauteur, largeur]
        if (size.ValueKind == JsonValueKind.Array) { h = size[0].GetInt32(); w = size[1].GetInt32(); } // liste : [hauteur, largeur]
        else { h = w = size.GetInt32(); } // nombre seul : carre
        string mode = Str(p, "interpolation", "auto"); // methode d'interpolation (calcul des nouveaux pixels)
        var flag = mode switch // traduit le nom en option OpenCV
        {
            "auto" => h * w < src.Rows * src.Cols ? InterpolationFlags.Area : InterpolationFlags.Linear, // Area si on reduit, sinon Linear
            "nearest" => InterpolationFlags.Nearest, // plus proche voisin (aucun melange de valeurs)
            "linear" => InterpolationFlags.Linear, // lineaire
            "cubic" => InterpolationFlags.Cubic, // cubique (plus lisse)
            "area" => InterpolationFlags.Area, // moyenne par zone (bon pour reduire)
            _ => throw new NotSupportedException($"interpolation inconnue : {mode}"), // nom inconnu : erreur
        };
        var dst = new Mat(); // image resultat
        Cv2.Resize(src, dst, new Size(w, h), 0, 0, flag); // redimensionne (Size prend largeur puis hauteur)
        return dst; // renvoie l'image redimensionnee
    }

    static Mat Crop(Mat src, Dictionary<string, JsonElement> p) // decoupe un rectangle ; renvoie le morceau
    {
        // coin haut-gauche et taille du rectangle (0 = jusqu'au bord)
        int x = (int)Num(p, "x", 0), y = (int)Num(p, "y", 0), width = (int)Num(p, "width", 0), height = (int)Num(p, "height", 0);
        int x2 = width <= 0 ? src.Cols : Math.Min(src.Cols, x + width); // bord droit, sans sortir de l'image
        int y2 = height <= 0 ? src.Rows : Math.Min(src.Rows, y + height); // bord bas, sans sortir de l'image
        if (x >= x2 || y >= y2) throw new ArgumentException("crop vide"); // rectangle vide : erreur
        return new Mat(src, new Rect(x, y, x2 - x, y2 - y)).Clone(); // vue sur le rectangle, copiee pour etre independante
    }

    static Mat GaussianBlur(Mat src, double sigma) // flou gaussien ; renvoie l'image floutee
    {
        var dst = new Mat(); // image resultat
        // Size(0, 0) : taille du noyau deduite de sigma ; bords traites en miroir (Reflect), comme en Python
        Cv2.GaussianBlur(src, dst, new Size(0, 0), sigma, 0, BorderTypes.Reflect);
        return dst; // renvoie l'image floutee
    }

    static Mat MedianBlur(Mat src, int ksize) // flou median ; renvoie l'image filtree
    {
        if (ksize % 2 == 0) ksize += 1; // la taille de fenetre doit etre impaire (un pixel au centre)
        if (src.Depth() != MatType.CV_8U && ksize > 5) // OpenCV ne gere le 16 bits que jusqu'a une fenetre de 5
            throw new NotSupportedException("median_blur 16 bits au-dela de 5 : la version Python passe par scipy"); // non porte : erreur
        var dst = new Mat(); // image resultat
        Cv2.MedianBlur(src, dst, ksize); // chaque pixel prend la mediane de son voisinage
        return dst; // renvoie l'image filtree
    }

    static Mat Clahe(Mat src, double clipLimit, int tileGrid) // CLAHE : egalisation de contraste par zones, limitee ; renvoie l'image
    {
        using var clahe = Cv2.CreateCLAHE(clipLimit, new Size(tileGrid, tileGrid)); // outil CLAHE, libere a la fin
        if (src.Channels() == 1) // image en gris
        {
            var dst = new Mat(); // image resultat
            clahe.Apply(src, dst); // applique CLAHE directement
            return dst; // renvoie l'image
        }
        // couleur 16 bits : pas encore porte
        if (src.Depth() != MatType.CV_8U) throw new NotSupportedException("clahe couleur 16 bits : a porter depuis contrast._on_luminance");
        // Couleur 8 bits : sur la luminance L de LAB, comme _on_luminance.
        var lab = src.CvtColor(ColorConversionCodes.RGB2Lab); // LAB : L = luminosite, A et B = couleur
        var planes = lab.Split(); // separe les 3 canaux L, A, B
        var l = new Mat(); // canal L corrige
        clahe.Apply(planes[0], l); // CLAHE sur la luminosite seulement (les couleurs ne bougent pas)
        planes[0] = l; // remplace L par sa version corrigee
        Cv2.Merge(planes, lab); // recolle les 3 canaux
        return lab.CvtColor(ColorConversionCodes.Lab2RGB); // repasse en RGB et renvoie
    }

    static Mat Gamma(Mat src, double gamma) // correction gamma : sortie = (entree / max) ^ gamma x max ; renvoie l'image
    {
        double peak = Peak(src); // 255 ou 65535
        int levels = (int)peak + 1; // nombre de valeurs possibles (256 ou 65536)
        if (src.Depth() == MatType.CV_8U) // image 8 bits
        {
            var table = new Mat(1, 256, MatType.CV_8U); // LUT (table de correspondance) : nouvelle valeur pour chacune des 256 valeurs
            // remplit la table ; arrondi au pair comme numpy
            for (int i = 0; i < 256; i++) table.Set(0, i, (byte)Math.Round(Math.Pow(i / peak, gamma) * peak, MidpointRounding.ToEven));
            var dst = new Mat(); // image resultat
            Cv2.LUT(src, table, dst); // remplace chaque pixel par sa valeur dans la table (tres rapide)
            return dst; // renvoie l'image corrigee
        }
        // 16 bits : table de 65536 entrees appliquee pixel par pixel (Cv2.LUT n'accepte que 8 bits).
        var lut = new ushort[levels]; // table de correspondance 16 bits
        for (int i = 0; i < levels; i++) lut[i] = (ushort)Math.Round(Math.Pow(i / peak, gamma) * peak, MidpointRounding.ToEven); // remplit
        var copy = src.Clone(); // copie de l'image (on ne modifie pas l'originale)
        var flat = copy.Reshape(1); // meme memoire, vue a 1 canal (GetArray<ushort> l'exige)
        flat.GetArray(out ushort[] data); // recupere les pixels dans un tableau C#
        for (int i = 0; i < data.Length; i++) data[i] = lut[data[i]]; // remplace chaque pixel par sa valeur dans la table
        flat.SetArray(data); // remet les pixels dans l'image
        return copy; // renvoie l'image corrigee
    }

    static Mat PercentileStretch(Mat src, double low, double high) // etire le contraste entre deux percentiles ; renvoie l'image
    {
        var x = AsFloat(src); // image en flottants dans [0, 1]
        var flat = x.Reshape(1, 1); // vue a plat : 1 canal, 1 ligne (tous les pixels a la suite)
        flat.GetArray(out float[] values); // recupere toutes les valeurs
        var sorted = (float[])values.Clone(); // copie a trier
        Array.Sort(sorted); // tri croissant (pour lire les percentiles)
        double lo = Percentile(sorted, low), hi = Percentile(sorted, high); // valeurs aux percentiles bas et haut
        var stretched = new Mat(); // image etiree
        // (x - lo) / (hi - lo) : lo devient 0 et hi devient 1 ; 1e-6 evite une division par zero
        x.ConvertTo(stretched, x.Type(), 1.0 / Math.Max(hi - lo, 1e-6), -lo / Math.Max(hi - lo, 1e-6));
        return FromFloat(stretched, src); // ecrete dans [0, 1] et revient au type d'origine
    }

    /// <summary>Percentile a interpolation lineaire, comme numpy.percentile par defaut.</summary>
    static double Percentile(float[] sorted, double q) // renvoie la valeur sous laquelle tombent q % des pixels
    {
        double rank = q / 100.0 * (sorted.Length - 1); // position (decimale) dans le tableau trie
        int below = (int)Math.Floor(rank); // case juste en dessous
        int above = Math.Min(below + 1, sorted.Length - 1); // case juste au-dessus (sans depasser la fin)
        return sorted[below] + (rank - below) * (sorted[above] - sorted[below]); // interpolation lineaire entre les deux
    }

    // correction d'eclairage : divise l'image par son fond lisse (flou tres large) ; renvoie l'image corrigee
    static Mat ShadingDivision(Mat src, double sigma, int downscale, bool keepMean, string method)
    {
        // seule la methode gaussienne est portee
        if (method != "gaussian") throw new NotSupportedException("shading_division median : a porter (scipy.ndimage.median_filter)");
        Mat[] channels = src.Channels() == 1 ? new[] { src } : src.Split(); // traite chaque canal separement
        var corrected = new Mat[channels.Length]; // canaux corriges
        for (int c = 0; c < channels.Length; c++) // pour chaque canal
        {
            var x = AsFloat(channels[c]); // canal en flottants dans [0, 1]
            int factor = Math.Max(1, downscale); // facteur de reduction (au moins 1) : flouter une petite image va plus vite
            var small = new Mat(); // version reduite
            // reduit l'image (Area : moyenne par zone)
            if (factor > 1) Cv2.Resize(x, small, new Size(Math.Max(1, x.Cols / factor), Math.Max(1, x.Rows / factor)), 0, 0, InterpolationFlags.Area);
            else small = x; // pas de reduction
            var blurred = new Mat(); // fond estime (version reduite)
            // flou tres large : ne garde que les variations lentes d'eclairage (sigma divise par le facteur de reduction)
            Cv2.GaussianBlur(small, blurred, new Size(0, 0), Math.Max(sigma / factor, 0.5), 0, BorderTypes.Reflect);
            var background = new Mat(); // fond a la taille d'origine
            if (factor > 1) Cv2.Resize(blurred, background, new Size(x.Cols, x.Rows), 0, 0, InterpolationFlags.Linear); // agrandit
            else background = blurred; // deja a la bonne taille
            var safe = new Mat(); // fond sans zero
            Cv2.Max(background, 1e-6, safe); // evite la division par zero
            var ratio = new Mat(); // image / fond
            Cv2.Divide(x, safe, ratio); // divise : un eclairage inegal devient uniforme (autour de 1)
            double gain = keepMean ? Cv2.Mean(background).Val0 : 0.5; // niveau final : luminosite moyenne d'origine, ou gris moyen (0.5)
            var scaled = new Mat(); // image remise a l'echelle
            ratio.ConvertTo(scaled, ratio.Type(), gain); // sans ce facteur l'image tournerait autour de 1 (blanc)
            corrected[c] = FromFloat(scaled, channels[c]); // retour au type d'origine (8 ou 16 bits)
        }
        if (corrected.Length == 1) return corrected[0]; // image grise : un seul canal a renvoyer
        var merged = new Mat(); // image couleur recomposee
        Cv2.Merge(corrected, merged); // recolle les canaux corriges
        return merged; // renvoie l'image corrigee
    }
}
