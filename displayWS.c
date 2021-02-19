#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include "include/handlers.h"
#include "include/iomat.h"
#include "include/imgproc3D.h"
#include "include/constants.hpp"
#include "include/imgproc3D.h"

using namespace std;
using namespace cv;
using namespace constants;

int main(int argc, char** argv)
{
	if(argc==1){
		cout<<endl;
		cout<<"------------------------------------"<<endl;
		cout<<"stack image display"<<endl;
		cout<<"------------------------------------"<<endl;
		cout<<"Giovanni Diana"<<endl<<endl;
		cout<<"usage:"<<endl;
		cout<<"./display -f <image>"<<endl;
		return 0;
	}

	ostringstream fname;
	int i,j,k;
	
	if(strcmp(argv[1],"-f")==0 && argv[2]!=NULL){
		fname<<argv[2];
	} else {
		cout<<"Missing filename"<<endl;
		return -1;
	}

    // Load the stack image
    Mat src = imread(fname.str().c_str(),IMREAD_UNCHANGED );
    MatIterator_<ushort> it;
    MatIterator_<Vec3b> it_C3;
	// check if the image is empty
	if(src.empty()){
		cout<<"Error: empty image."<<endl;
		return -1;
	}

	// Construct the 3D image as a vector of 2D images.
	vector<Mat> image3D(nlayers);	// image to show.
	
	vector<Mat> mer(nlayers);
    for(i=0;i<nlayers;i++) mer[i]=Mat::zeros(hNROWS,NCOLS,CV_8UC3);
    
    for(i=0;i<nlayers;i++){
		image3D[i]=src(Range(hNROWS*i,hNROWS+hNROWS*i),Range::all());
	}

	// calculate the maximum fluorescence value.
	ushort orig_max=max3D(image3D);
    RNG rng;
    vector< Vec3b > rc(orig_max);

    for(i=1;i<=orig_max;i++){
        for(j=0;j<3;j++) rc[i-1][j]=rng.uniform(0,255);
    }        

    for(j=1;j<=orig_max;j++){
        for(k=0;k<nlayers;k++){
            it_C3=mer[k].begin<Vec3b>();
            for(it=image3D[k].begin<ushort>();it!=image3D[k].end<ushort>();++it){
                if(*it==j) *it_C3=rc[j-1];
                ++it_C3;
            }
        }
    }

    //for(i=0;i<nlayers;i++){
	//	mer[i].convertTo(mer[i],CV_32FC3);
	//}
    
    // Create the window to display the 3D image and the WST
	//cvNamedWindow( "Display window", WINDOW_NORMAL | WINDOW_KEEPRATIO ); // Create a window for display.
	cvNamedWindow( "Display window", WINDOW_KEEPRATIO ); // Create a window for display.
	resizeWindow( "Display window", NCOLS,hNROWS);

	int brightness=7;
  
    // Set up of the GUI.
	int level=0;
    //for(i=0;i<nlayers;i++) mer[i].convertTo(mer[i],CV_32FC3,1.0/255);
	cvCreateTrackbar( "Layer", "Display window", &level, 50,  NULL);
	cvCreateTrackbar( "Brightnessr", "Display window", &brightness, 50,  NULL);
	imshow("Display window",mer[level]);
    Point2i param(0,0);

    while(cvWaitKey(30) != 'q')	 {
	     cvSetMouseCallback("Display window",on_mouse_disp,&param);
		 ostringstream disp;
		 disp<<"x="<<param.y<<", y="<<param.x<<", value="<<src.at<ushort>(hNROWS*level+param.y,param.x);
		 displayStatusBar("Display window", disp.str(), 0 );
		 imshow("Display window",mer[level]);

	}

	cvDestroyAllWindows();
	return 0;

}

