#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include "include/handlers.h"
#include "include/iomat.h"
#include "include/imgproc3D.h"
#include "include/flu_fun.h"
#include "include/constants.hpp"
#include <sys/stat.h>

using namespace std;
using namespace cv;
using namespace constants;


int main(int argc, char** argv)
{
	if(argc==1){
		cout<<endl;
		cout<<"------------------------------------"<<endl;
		cout<<"Watershed algorithm for stack images"<<endl;
		cout<<"Automated version.                  "<<endl;
		cout<<"------------------------------------"<<endl;
		cout<<"Giovanni Diana"<<endl<<endl;
		cout<<"usage:"<<endl;
		cout<<"./autows3D <image> <ID> <proc_folder> <tolerance> <radius> <MAX_ASSIGNMENT>"<<endl;
		return 0;
	}

	ostringstream fname;
	int i;

	fname<<argv[1];
	char* ID=argv[2]; 
	char* proc_folder=argv[3];

	stringstream daf7_WS,ins6_WS;
	daf7_WS<<proc_folder<<'/'<<"ID"<<ID<<"daf7_WS2.tiff";
	ins6_WS<<proc_folder<<'/'<<"ID"<<ID<<"ins6_WS2.tiff";
	
	struct stat sb;

	if (stat(proc_folder, &sb) != 0){
		const int dir_err = mkdir(proc_folder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (-1 == dir_err){
			printf("Error creating directory!");
			exit(1);
		}
	}


	int tol=atoi(argv[4]);
	int rad=atoi(argv[5]);
	int MAX_ASSIGNMENT=atoi(argv[6]);
	double ins6th= (argc == 8 ) ? (double)atoi(argv[7]) : 1.0;
	Point3i Pt_tmp;

	cout<<"WORM ID = "<<ID<<endl;

/* 
    Mat objects:
    ----------------------------------------------------------------------------------------

    NAME          TYPE            DESCRIPTION
    ----------------------------------------------------------------------------------------
	src  		| ushort	   |  is the source stack image of 51 layers. 
		 		|        	   |  it is always stored as CV_16U

	top_image3D | ushort 	   |  these matrices are vector<Mat> where each element is Mat<ushort>
	bot_image3D | ushort 	   |

    top_gs      | ushort,float |  These are the filtered images, and the watershed transforms in 3D.  
	bot_gs      |              |  These are shown in "Display Window". Initially stored as ushort and 
	top_wst     |              |  then converted into float to be able to normalize them and copied 
    bot_wst     |              |  onto image3D.

	top_NeurVec | uchar        |  This are the masks representing each neuron. Both are defined as
	bot_NeurVec |    		   |  vector<vector<Mat> > i.e. a vector (of length 6=N_neur) of 3D images 
    ---------------------------------------------------------------------------------------------

	Iterators
	---------------------------------------

    NAME         TYPE             DESCRIPTION
    ----------------------------------------------------------------------------------------
    it          | ushort       |  Iterator over top_ and bot_ image3D[k]
	n_it        | char         |  Iterate over top_ and bot_ NeurVec.

*/

    // Load the stack image
    const Mat src = imread(fname.str().c_str(),IMREAD_UNCHANGED );
	// check if the image is empty
	if(src.empty()){
		cout<<"Error: empty image."<<endl;
		return -1;
	}

	// Construct the 3D image as a vector of 2D images.
	vector<Mat> top_image3D(nlayers);
	vector<Mat> bot_image3D(nlayers);
	vector<Mat> top_image3D_float(nlayers);
	vector<Mat> bot_image3D_float(nlayers);
	vector<Mat> top_gs(nlayers); 	// gaussian filtered top image
	vector<Mat> bot_gs(nlayers); 	// gaussian filtered bot image
	vector<Mat> top_gs_float(nlayers); 	// gaussian filtered top image
	vector<Mat> bot_gs_float(nlayers); 	// gaussian filtered bot image
	vector<Mat> top_wst(nlayers);  	// matrix to store the watershed transform (top)
	vector<Mat> bot_wst(nlayers);  	// matrix to store the watershed transform (bot)
	Mat top_wst_flat=Mat::zeros(hNROWS*nlayers,NCOLS,CV_16U);
	Mat bot_wst_flat=Mat::zeros(hNROWS*nlayers,NCOLS,CV_16U);

    // Matrix for dilation operation.
	int dilation_size=12;
	Mat element = getStructuringElement(MORPH_ELLIPSE, 
	                    Size( 2*dilation_size + 1, 2*dilation_size+1 ),
						Point( dilation_size, dilation_size ) );
    
	// Initialization of output images
	for(i=0;i<nlayers;i++){
		// Extract top and bottom images from the source.
		top_image3D[i]=src(Range(NROWS*i,hNROWS+NROWS*i),Range::all());
		bot_image3D[i]=src(Range(hNROWS+NROWS*i,NROWS+NROWS*i),Range::all());
		// initialize watershed matrix
		top_wst[i] = Mat::zeros(hNROWS,NCOLS,CV_16U);
		bot_wst[i] = Mat::zeros(hNROWS,NCOLS,CV_16U);
		// initialize gaussian filter matrices
		top_gs[i] = Mat::zeros(hNROWS,NCOLS,CV_16U);
		bot_gs[i] = Mat::zeros(hNROWS,NCOLS,CV_16U);
	}
	
	// Run the filter and calculate the maximum fluorescence value.
	Blur3D(top_image3D,top_gs,5,1);
	Blur3D(bot_image3D,bot_gs,5,1);
	ushort top_max=max3D(top_gs);
	ushort bot_max=max3D(bot_gs);
	ushort top_meanflu=mean3D(top_gs);
	ushort bot_meanflu=mean3D(bot_gs);
	ushort top_sd=var3D(top_gs,(float)top_meanflu);
	ushort bot_sd=var3D(bot_gs,(float)bot_meanflu);
	ushort top_minflu=top_meanflu+0.5*top_sd;
	ushort bot_minflu=bot_meanflu+ins6th*bot_sd;

	// Calculate the watershed transform in 3D.
	cout<<"---------------------------------------"<<endl;
	cout<<"WATERSHED "<<endl;
	watershed3D(top_gs,top_wst,(ushort)top_max,top_minflu,tol,rad,MAX_ASSIGNMENT);
	watershed3D(bot_gs,bot_wst,(ushort)bot_max,bot_minflu,tol,rad,MAX_ASSIGNMENT);
	cout<<"---------------------------------------"<<endl;
	for(i=0;i<nlayers;i++) {
		top_wst[i].copyTo(top_wst_flat(Range(hNROWS*i,hNROWS*i+hNROWS),Range::all()));
		bot_wst[i].copyTo(bot_wst_flat(Range(hNROWS*i,hNROWS*i+hNROWS),Range::all()));
	}
	imwrite(daf7_WS.str(),top_wst_flat);
	imwrite(ins6_WS.str(),bot_wst_flat);

		
	return 0;

}

