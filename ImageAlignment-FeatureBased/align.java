import java.util.*;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.ORB;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Point;

public class App {

    static int MAX_FEATURES = 500;
    static float GOOD_MATCH_PERCENT = 0.15f;
    static String imagePath = "path to your image folder";

    static {
        nu.pattern.OpenCV.loadShared();
        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Read reference image
        String refFilename = imagePath + "form.jpg";
        System.out.println("Reading reference image : " + refFilename);
        Mat imReference = Imgcodecs.imread(refFilename);

        // Read image to be aligned
        String imFilename = imagePath + "scanned-form.jpg";
        System.out.println("Reading image to align : " + imFilename);
        Mat im = Imgcodecs.imread(imFilename);

        // Registered image will be resotred in imReg.
        // The estimated homography will be stored in h.
        Mat imReg = new Mat();
        Mat h = new Mat();

        // Align images
        System.out.println("Aligning images ...");
        alignImages(im, imReference, imReg, h);

        // Write aligned image to disk.
        String outFilename = imagePath + "aligned.jpg";
        System.out.println("Saving aligned image : " + outFilename);
        Imgcodecs.imwrite(outFilename, imReg);

        System.out.println("Estimated homography :" + h);
    }

    static Comparator<DMatch> ascOrder = new Comparator<DMatch>() {
        public int compare(DMatch arg0, DMatch arg1) {
            return Double.compare(arg0.distance, arg1.distance);
        }
    };

    static void alignImages(Mat im1, Mat im2, Mat im1Reg, Mat h) {
        // Convert images to grayscale
        Mat im1Gray = new Mat();
        Mat im2Gray = new Mat();
        Imgproc.cvtColor(im1, im1Gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.cvtColor(im2, im2Gray, Imgproc.COLOR_BGR2GRAY);

        // Variables to store keypoints and descriptors
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();

        // Detect ORB features and compute descriptors.
        Feature2D orb = ORB.create();
        orb.detectAndCompute(im1Gray, new Mat(), keypoints1, descriptors1);
        orb.detectAndCompute(im2Gray, new Mat(), keypoints2, descriptors2);

        // Match features.
        MatOfDMatch matches = new MatOfDMatch();
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        matcher.match(descriptors1, descriptors2, matches, new Mat());

        List<DMatch> lmatches = matches.toList();
        List<DMatch> goodMatches = new ArrayList<>();

        // Sort matches by score(distance)
        lmatches.sort(ascOrder);

        int numGoodMatches = Math.round(lmatches.size() * GOOD_MATCH_PERCENT);

        // consider only good matches
        int index = 0;
        while (index < numGoodMatches) {
            goodMatches.add(lmatches.get(index));
            index = index + 1;
        }

        matches.fromList(goodMatches);

        // Draw top matches
        Mat imMatches = new Mat();
        Features2d.drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
        Imgcodecs.imwrite(imagePath + "matches.jpg", imMatches);

        List<KeyPoint> key1 = keypoints1.toList();
        List<KeyPoint> key2 = keypoints2.toList();

        // Extract location of good matches
        List<Point> p1 = new ArrayList<>();
        List<Point> p2 = new ArrayList<>();

        for (DMatch m : goodMatches) {
            p1.add(key1.get(m.queryIdx).pt);
            p2.add(key2.get(m.trainIdx).pt);
        }

        MatOfPoint2f points1 = new MatOfPoint2f();
        MatOfPoint2f points2 = new MatOfPoint2f();

        points1.fromList(p1);
        points2.fromList(p2);

        // Find homography
        double ransacReprojThreshold = 3.0;
        h = Calib3d.findHomography(points1, points2, Calib3d.RANSAC, ransacReprojThreshold);

        // Use homography to warp image
        Imgproc.warpPerspective(im1, im1Reg, h, im2.size());
    }
}
