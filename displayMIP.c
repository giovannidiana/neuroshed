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
		cout<<"./display <ID_image>"<<endl;
		return 0;
	}

	ostringstream fname;
	int i;
	//char* ID=argc >= 2 ? argv[1] : (char*) 1; 

	//string main_dir="/home/diana/workspace/data/dataset-add-2";
	//string folder="20140922-20140916_196_29-d6.0";
	//string source="20140916_196_29";
	//string main_dir="/home/diana/workspace/data/data-June_WT";
	//string folder="20140621-20140615_196_SB-d6.0";
	//string source="20140615_196_SB";
	//fname<<main_dir<<"/";
	//fname<<folder<<"/";
	//fname<<source;
	//fname<<"-d6.0xAutoStack"<<ID<<".tiff";
	
	fname<<argv[1];

    // Load the stack image
    Mat src = imread(fname.str().c_str(),IMREAD_UNCHANGED );

    Mat top_mip=Mat::zeros(hNROWS,NCOLS,CV_16U);
	Mat bot_mip=Mat::zeros(hNROWS,NCOLS,CV_16U);

	// check if the image is empty
	if(src.empty()){
		cout<<"Error: empty image."<<endl;
		return -1;
	}

	// Construct the 3D image as a vector of 2D images.
	vector<Mat> top_image3D(nlayers);
	vector<Mat> bot_image3D(nlayers);
	Mat image3D;	// image to show.
    
    for(i=0;i<nlayers;i++){
		top_image3D[i]=src(Range(NROWS*i,hNROWS+NROWS*i),Range::all());
		bot_image3D[i]=src(Range(hNROWS+NROWS*i,NROWS+NROWS*i),Range::all());
	}
	image3D= Mat::zeros(NROWS,NCOLS,CV_32F);

	cout<<"NCOL = "<<top_image3D[0].cols<<endl;
	
	// calculate the maximum fluorescence value.
	ushort top_orig_max=max3D(top_image3D);
	ushort bot_orig_max=max3D(bot_image3D);
	
	// Calculate MIP
	MIP(top_image3D,top_mip);
	MIP(bot_image3D,bot_mip);

	// Convert & Normalize MIPs
	top_mip.convertTo(top_mip,CV_32F,1./top_orig_max);
	bot_mip.convertTo(bot_mip,CV_32F,1./bot_orig_max);

	cvNamedWindow( "Display window", WINDOW_FREERATIO ); // Create a window for display.
	resizeWindow( "Display window", NCOLS,NROWS);
	moveWindow("Display window",NCOLS*2,0);

	int brightness=1;
  
    // Copy to image to display 
	top_mip.copyTo(image3D(Range(0,hNROWS),Range::all()));
	bot_mip.copyTo(image3D(Range(hNROWS,NROWS),Range::all()));

    // Set up of the GUI.
	cvCreateTrackbar( "Brightnessr", "Display window", &brightness, 50,  NULL);
	imshow("Display window",brightness/20.*image3D);
    Point2i param(0,0);

    while(cvWaitKey(30) != 'q')	 {
	     cvSetMouseCallback("Display window",on_mouse_disp,&param);
		 ostringstream disp;
		 disp<<"x="<<param.y<<", y="<<param.x<<", value="<<src.at<ushort>(param.y,param.x);
		 displayStatusBar("Display window", disp.str(), 0 );
		 imshow("Display window",20/brightness*image3D);

	}

	cvDestroyAllWindows();
	return 0;

}

