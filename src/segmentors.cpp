#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Function to perform Hierarchical Clustering-based image segmentation
Mat performHierarchicalClustering(const Mat& image) {
    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise
    Mat blurredImage;
    GaussianBlur(grayImage, blurredImage, Size(15, 15), 0);

    // Reshape the image to be a single column of 32-bit floating point values
    Mat reshapedImage;
    blurredImage.convertTo(reshapedImage, CV_32F);
    reshapedImage = reshapedImage.reshape(1, reshapedImage.total());

    // Perform Hierarchical Clustering
    Mat labels;
    int numClusters = 2;  // Adjust the number of clusters as needed
    TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 0.1);
    int attempts = 20;
    int flags = KMEANS_RANDOM_CENTERS;
    kmeans(reshapedImage, numClusters, labels, criteria, attempts, flags);

    // Reshape the labels and convert to 8-bit image
    labels = labels.reshape(0, blurredImage.rows);
    labels.convertTo(labels, CV_8U);

    double minVal; 
    double maxVal; 
    Point minLoc; 
    Point maxLoc;
    cv::minMaxLoc( labels, &minVal, &maxVal, &minLoc, &maxLoc );

    labels *= maxVal / 255;
    return labels;
}

// 5-Dimensional Point
class Point5D{
	public:
		float x;			// Spatial value
		float y;			// Spatial value
		float l;			// Lab value
		float a;			// Lab value
		float b;			// Lab value
	public:
		Point5D();													// Constructor
		~Point5D();													// Destructor
		void PointLab();											// Scale the OpenCV Lab color to Lab range
		void PointRGB();											// Sclae the Lab color to OpenCV range that can be used to transform to RGB
		void MSPoint5DAccum(Point5D);								// Accumulate points
		void MSPoint5DCopy(Point5D);								// Copy a point
		float MSPoint5DColorDistance(Point5D);						// Compute color space distance between two points
		float MSPoint5DSpatialDistance(Point5D);					// Compute spatial space distance between two points
		void MSPoint5DScale(float);									// Scale point
		void MSPOint5DSet(float, float, float, float, float);		// Set point value
		void Print();												// Print 5D point
};

class MeanShift{
	public:
		float hs;				// spatial radius
		float hr;				// color radius
		vector<Mat> IMGChannels;
	public:
		MeanShift(float, float);									// Constructor for spatial bandwidth and color bandwidth
		void MSFiltering(Mat&);										// Mean Shift Filtering
		void MSSegmentation(Mat&);									// Mean Shift Segmentation
};


#define MS_MAX_NUM_CONVERGENCE_STEPS	5										// up to 10 steps are for convergence
#define MS_MEAN_SHIFT_TOL_COLOR			0.3										// minimum mean color shift change
#define MS_MEAN_SHIFT_TOL_SPATIAL		0.3										// minimum mean spatial shift change
const int dxdy[][2] = {{-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1}};	// Region Growing

// Constructor
Point5D::Point5D(){
	x = -1;
	y = -1;
}

// Destructor
Point5D::~Point5D(){
}

// Scale the OpenCV Lab color to Lab range
void Point5D::PointLab(){
	l = l * 100 / 255;
	a = a - 128;
	b = b - 128;
}

// Sclae the Lab color to OpenCV range that can be used to transform to RGB
void Point5D::PointRGB(){
	l = l * 255 / 100;
	a = a + 128;
	b = b + 128;
}

// Accumulate points
void Point5D::MSPoint5DAccum(Point5D Pt){
	x += Pt.x;
	y += Pt.y;
	l += Pt.l;
	a += Pt.a;
	b += Pt.b;
}

// Copy a point
void Point5D::MSPoint5DCopy(Point5D Pt){
	x = Pt.x;
	y = Pt.y;
	l = Pt.l;
	a = Pt.a;
	b = Pt.b;
}

// Compute color space distance between two points
float Point5D::MSPoint5DColorDistance(Point5D Pt){
	return sqrt((l - Pt.l) * (l - Pt.l) + (a - Pt.a) * (a - Pt.a) + (b - Pt.b) * (b - Pt.b));
}

// Compute spatial space distance between two points
float Point5D::MSPoint5DSpatialDistance(Point5D Pt){
	return sqrt((x - Pt.x) * (x - Pt.x) + (y - Pt.y) * (y - Pt.y));
}

// Scale point
void Point5D::MSPoint5DScale(float scale){
	x *= scale;
	y *= scale;
	l *= scale;
	a *= scale;
	b *= scale;
}

// Set point value
void Point5D::MSPOint5DSet(float px, float py, float pl, float pa, float pb){
	x = px;
	y = py;
	l = pl;
	a = pa;
	b = pb;
}

// Print 5D point
void Point5D::Print(){
	cout<<x<<" "<<y<<" "<<l<<" "<<a<<" "<<b<<endl;
}

// Constructor for spatial bandwidth and color bandwidth
MeanShift::MeanShift(float s, float r){
	hs = s;
	hr = r;
}

// Mean Shift Filtering
void MeanShift::MSFiltering(Mat& Img){
	int ROWS = Img.rows;			// Get row number
	int COLS = Img.cols;			// Get column number
	split(Img, IMGChannels);		// Split Lab color

	Point5D PtCur;					// Current point
	Point5D PtPrev;					// Previous point
	Point5D PtSum;					// Sum vector of the shift vector
	Point5D Pt;
	int Left;						// Left boundary
	int Right;						// Right boundary
	int Top;						// Top boundary
	int Bottom;						// Bottom boundary
	int NumPts;						// number of points in a hypersphere
	int step;
	
	for(int i = 0; i < ROWS; i++){
		for(int j = 0; j < COLS; j++){
			Left = (j - hs) > 0 ? (j - hs) : 0;						// Get Left boundary of the filter
			Right = (j + hs) < COLS ? (j + hs) : COLS;				// Get Right boundary of the filter
			Top = (i - hs) > 0 ? (i - hs) : 0;						// Get Top boundary of the filter
			Bottom = (i + hs) < ROWS ? (i + hs) : ROWS;				// Get Bottom boundary of the filter
			// Set current point and scale it to Lab color range
			PtCur.MSPOint5DSet(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
			PtCur.PointLab();
			step = 0;				// count the times
			do{
				PtPrev.MSPoint5DCopy(PtCur);						// Set the original point and previous one
				PtSum.MSPOint5DSet(0, 0, 0, 0, 0);					// Initial Sum vector
				NumPts = 0;											// Count number of points that satisfy the bandwidths
				for(int hx = Top; hx < Bottom; hx++){
					for(int hy = Left; hy < Right; hy++){
						// Set point in the spatial bandwidth
						Pt.MSPOint5DSet(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
						Pt.PointLab();

						// Check it satisfied color bandwidth or not
						if(Pt.MSPoint5DColorDistance(PtCur) < hr){
							PtSum.MSPoint5DAccum(Pt);				// Accumulate the point to Sum vector
							NumPts++;								// Count
						}
					}
				}
				PtSum.MSPoint5DScale(1.0 / NumPts);					// Scale Sum vector to average vector
				PtCur.MSPoint5DCopy(PtSum);							// Get new origin point
				step++;												// One time end
			// filter iteration to end
			}while((PtCur.MSPoint5DColorDistance(PtPrev) > MS_MEAN_SHIFT_TOL_COLOR) && (PtCur.MSPoint5DSpatialDistance(PtPrev) > MS_MEAN_SHIFT_TOL_SPATIAL) && (step < MS_MAX_NUM_CONVERGENCE_STEPS));
			
			// Scale the color
			PtCur.PointRGB();
			// Copy the result to image
			Img.at<Vec3b>(i, j) = Vec3b(PtCur.l, PtCur.a, PtCur.b);
		}
	}
}

void MeanShift::MSSegmentation(Mat& Img){

//---------------- Mean Shift Filtering -----------------------------
	// Same as MSFiltering function
	int ROWS = Img.rows;
	int COLS = Img.cols;
	split(Img, IMGChannels);

	Point5D PtCur;
	Point5D PtPrev;
	Point5D PtSum;
	Point5D Pt;
	int Left;
	int Right;
	int Top;
	int Bottom;
	int NumPts;					// number of points in a hypersphere
	int step;
	
	for(int i = 0; i < ROWS; i++){
		for(int j = 0; j < COLS; j++){
			Left = (j - hs) > 0 ? (j - hs) : 0;
			Right = (j + hs) < COLS ? (j + hs) : COLS;
			Top = (i - hs) > 0 ? (i - hs) : 0;
			Bottom = (i + hs) < ROWS ? (i + hs) : ROWS;
			PtCur.MSPOint5DSet(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
			PtCur.PointLab();
			step = 0;
			do{
				PtPrev.MSPoint5DCopy(PtCur);
				PtSum.MSPOint5DSet(0, 0, 0, 0, 0);
				NumPts = 0;
				for(int hx = Top; hx < Bottom; hx++){
					for(int hy = Left; hy < Right; hy++){
						
						Pt.MSPOint5DSet(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
						Pt.PointLab();

						if(Pt.MSPoint5DColorDistance(PtCur) < hr){
							PtSum.MSPoint5DAccum(Pt);
							NumPts++;
						}
					}
				}
				PtSum.MSPoint5DScale(1.0 / NumPts);
				PtCur.MSPoint5DCopy(PtSum);
				step++;
			}while((PtCur.MSPoint5DColorDistance(PtPrev) > MS_MEAN_SHIFT_TOL_COLOR) && (PtCur.MSPoint5DSpatialDistance(PtPrev) > MS_MEAN_SHIFT_TOL_SPATIAL) && (step < MS_MAX_NUM_CONVERGENCE_STEPS));
			
			PtCur.PointRGB();
			Img.at<Vec3b>(i, j) = Vec3b(PtCur.l, PtCur.a, PtCur.b);
		}
	}
//--------------------------------------------------------------------

//----------------------- Segmentation ------------------------------
	int RegionNumber = 0;			// Reigon number
	int label = -1;					// Label number
	float *Mode = new float [ROWS * COLS * 3];					// Store the Lab color of each region
	int *MemberModeCount = new int [ROWS * COLS];				// Store the number of each region
	memset(MemberModeCount, 0, ROWS * COLS * sizeof(int));		// Initialize the MemberModeCount
	split(Img, IMGChannels);
	// Label for each point
	int **Labels = new int *[ROWS];
	for(int i = 0; i < ROWS; i++)
		Labels[i] = new int [COLS];

	// Initialization
	for(int i = 0; i < ROWS; i++){
		for(int j = 0; j < COLS; j++){
			Labels[i][j] = -1;
		}
	}

	for(int i = 0; i < ROWS; i++){
		for(int j = 0; j < COLS;j ++){
			// If the point is not being labeled
			if(Labels[i][j] < 0){
				Labels[i][j] = ++label;		// Give it a new label number
				// Get the point
				PtCur.MSPOint5DSet(i, j, (float)IMGChannels[0].at<uchar>(i, j), (float)IMGChannels[1].at<uchar>(i, j), (float)IMGChannels[2].at<uchar>(i, j));
				PtCur.PointLab();

				// Store each value of Lab
				Mode[label * 3 + 0] = PtCur.l;
				Mode[label * 3 + 1] = PtCur.a;
				Mode[label * 3 + 2] = PtCur.b;
				
				// Region Growing 8 Neighbours
				vector<Point5D> NeighbourPoints;
				NeighbourPoints.push_back(PtCur);
				while(!NeighbourPoints.empty()){
					Pt = NeighbourPoints.back();
					NeighbourPoints.pop_back();

					// Get 8 neighbours
					for(int k = 0; k < 8; k++){
						int hx = Pt.x + dxdy[k][0];
						int hy = Pt.y + dxdy[k][1];
						if((hx >= 0) && (hy >= 0) && (hx < ROWS) && (hy < COLS) && (Labels[hx][hy] < 0)){
							Point5D P;
							P.MSPOint5DSet(hx, hy, (float)IMGChannels[0].at<uchar>(hx, hy), (float)IMGChannels[1].at<uchar>(hx, hy), (float)IMGChannels[2].at<uchar>(hx, hy));
							P.PointLab();

							// Check the color
							if(PtCur.MSPoint5DColorDistance(P) < hr){
								// Satisfied the color bandwidth
								Labels[hx][hy] = label;				// Give the same label					
								NeighbourPoints.push_back(P);		// Push it into stack
								MemberModeCount[label]++;			// This region number plus one
								// Sum all color in same region
								Mode[label * 3 + 0] += P.l;
								Mode[label * 3 + 1] += P.a;
								Mode[label * 3 + 2] += P.b;
							}
						}
					}
				}
				MemberModeCount[label]++;							// Count the point itself
				Mode[label * 3 + 0] /= MemberModeCount[label];		// Get average color
				Mode[label * 3 + 1] /= MemberModeCount[label];
				Mode[label * 3 + 2] /= MemberModeCount[label];
			}
		}
	}
	RegionNumber = label + 1;										// Get region number
	
	// Get result image from Mode array
	for(int i = 0; i < ROWS; i++){
		for(int j = 0; j < COLS; j++){
			label = Labels[i][j];
			float l = Mode[label * 3 + 0];
			float a = Mode[label * 3 + 1];
			float b = Mode[label * 3 + 2];
			Point5D Pixel;
			Pixel.MSPOint5DSet(i, j, l, a, b);
			Pixel.PointRGB();
//			Pixel.Print();
			Img.at<Vec3b>(i, j) = Vec3b(Pixel.l, Pixel.a, Pixel.b);
		}
	}
//--------------------------------------------------------------------

//	for(int i = 0; i < ROWS; i++){
//		for(int j = 0; j < COLS - 1; j++){
//			if(Labels[i][j] != Labels[i][j + 1])
//				Img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
//		}
//	}

//--------------- Delete Memory Applied Before -----------------------
	delete[] Mode;
	delete[] MemberModeCount;
	
	for(int i = 0; i < ROWS; i++)
		delete[] Labels[i];
	delete[] Labels;
}

Mat performMeanShiftClustering(const Mat& image, float spatialRadius, float colorRadius) {
    // Convert the image to the Lab color space
    Mat labImage;
    cvtColor(image, labImage, COLOR_BGR2Lab);

    // Reshape the image to be a 2D matrix of floating point values
    Mat reshapedImage;
    labImage.convertTo(reshapedImage, CV_32FC3);
    reshapedImage = reshapedImage.reshape(1, reshapedImage.total());

    // Perform Mean Shift clustering
    Mat labels;
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0);
    Rect roi = Rect(0, 0, 100, 100);
    meanShift(reshapedImage, roi, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 1));

    // Reshape the labels and convert to 8-bit image
    // labels = labels.reshape(0, labImage.rows);

    labels.convertTo(labels, CV_8U);

    return labels;
}

int main() {
    Mat image = imread("../data/tray2/food_image.jpg");
    if (image.empty()) {
        cout << "Failed to load image." << endl;
        return -1;
    }

	Mat smoothImage;
	// int i = 10;
	GaussianBlur(image, smoothImage, Size(31, 31), 0);
	// bilateralFilter(image, smoothImage, i, i*2, i/2);
	imshow("SS", smoothImage);
	waitKey(0);

    // Convert the image to grayscale
    Mat grayImage;	
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Apply thresholding to obtain a binary image
    Mat binaryImage;
    threshold(grayImage, binaryImage, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // Perform morphological operations to remove noise and close gaps
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(binaryImage, binaryImage, MORPH_OPEN, kernel, Point(-1, -1), 2);
    dilate(binaryImage, binaryImage, kernel, Point(-1, -1), 2);

    // Perform the watershed algorithm
    Mat markers;
    distanceTransform(binaryImage, markers, DIST_L2, 5);
    normalize(markers, markers, 0, 255, NORM_MINMAX);
    markers.convertTo(markers, CV_32SC1);  // Convert markers to the required type

    // Apply the watershed algorithm
    watershed(smoothImage, markers);

    // Generate random colors for visualization
    vector<Vec3b> colors;
    for (int i = 0; i < 3; i++) {
        int b = theRNG().uniform(0, 256);
        int g = theRNG().uniform(0, 256);
        int r = theRNG().uniform(0, 256);
        colors.push_back(Vec3b(b, g, r));
    }
	// cout << markers << endl;
    // Create a colored output image based on the watershed result
    Mat outputImage = Mat::zeros(image.size(), CV_8UC3);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            int index = markers.at<int>(i, j);
            if (index > 0 && index <= colors.size())
                outputImage.at<Vec3b>(i, j) = colors[index - 1];
            else
                outputImage.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
        }
    }

    // Display the original image and the watershed result
    namedWindow("Original Image", WINDOW_NORMAL);
    namedWindow("Watershed Result", WINDOW_NORMAL);
    imshow("Original Image", image);
    imshow("Watershed Result", outputImage);
    waitKey(0);

    return 0;
}