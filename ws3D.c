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
#include <GL/glut.h>

using namespace std;
using namespace cv;
using namespace constants;


int main(int argc, char** argv)
{
    glutInit( & argc, argv );
	ofstream lock_file("lock");
	lock_file<<1;
	lock_file.close();


	if(argc==1){
		cout<<endl;
		cout<<"------------------------------------"<<endl;
		cout<<"Watershed algorithm for stack images"<<endl;
		cout<<"version: 1.2 (15.02.2016)"<<endl;
		cout<<"------------------------------------"<<endl;
		cout<<"Giovanni Diana"<<endl<<endl;
		cout<<"usage:"<<endl;
		cout<<"./ws3D <image> <ID> <proc_folder> <tolerance> <radius> <MAX_ASSIGNMENT> <DOWS=0>"<<endl;
		return 0;
	}

	ostringstream fname;
	int i,k,kp;
	int ip,jp;

    fname<<argv[1];
    char* ID=argv[2];
	char* proc_folder=argv[3];
	stringstream winname;
	winname<<"Display window - worm "<<ID;

	stringstream daf7_WS,ins6_WS,logfilename;
	bool DOWS= (argc == 8 && strcmp(argv[7],"DOWS")==0) ? true : false;
	daf7_WS<<proc_folder<<'/'<<"ID"<<ID<<"daf7_WS2.tiff";
	ins6_WS<<proc_folder<<'/'<<"ID"<<ID<<"ins6_WS2.tiff";
	logfilename<<proc_folder<<'/'<<"ID"<<ID<<"_2.log";

	ofstream logfile(logfilename.str().c_str());

	
	struct stat sb;

	if (stat(proc_folder, &sb) != 0){
		const int dir_err = mkdir(proc_folder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		if (-1 == dir_err){
			printf("Error creating directory!");
			exit(1);
		}
	}


	ostringstream resfilename;
	resfilename<<proc_folder<<"/ID"<<ID<<"_2.dat";
	ofstream resfile(resfilename.str().c_str());

	int tol=atoi(argv[4]);
	int rad=atoi(argv[5]);
	int MAX_ASSIGNMENT=atoi(argv[6]);
	int top_FL=0,top_BG=0;
	int bot_FL=0,bot_BG=0;
	Point3i Pt_tmp;
	Mat Mat_tmp,mask_tmp,mask_tmp_float;
	vector<Mat> split(3);
	MatIterator_<ushort> it;
	vector<MatIterator_<uchar> > n_it(N_neur);
	for(i=0;i<3;i++) split[i]=Mat::zeros(hNROWS,NCOLS,CV_32F);
	Mat mer(hNROWS,NCOLS,CV_32FC3);

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

	Mat_tmp     | ushort 	   |  These are temporary matrices used to crop sub images to calculate
	mask_tmp    | float    	   |  the background. Mat_tmp is directly obtained by cutting top_image3D 
	            |        	   |  and bot_image3D while mask is a float image since is obtained from 
				|		 	   |  the watershed images.

    top_gs      | ushort,float |  These are the filtered images, and the watershed transforms in 3D.  
	bot_gs      |              |  These are shown in "Display Window". Initially stored as ushort and 
	top_wst     |              |  then converted into float to be able to normalize them and copied 
    bot_wst     |              |  onto image3D.

	image3D     | float        |  This is a vector<Mat> with float entries and size NROWSxNCOLS.
	            |              |  the top and bottom part after conversion and normalization are copied
				|              |  here to be displayed.

	top_mip     | ushort,float |  top and bot mip are the Maximum intensity projections of top_ and bot_
	bot_mip     | ushort,float |  image3D. Initially ushort are converted into float then normalized 
	mip         | float        |  and copied onto mip which is then displayed.

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
	cout<<"Source image loaded"<<endl;

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
	vector<Mat> image3D(nlayers);	// image to show.
	Mat top_mip=Mat::zeros(hNROWS,NCOLS,CV_16U);
	Mat bot_mip=Mat::zeros(hNROWS,NCOLS,CV_16U);
	Mat mip=Mat::zeros(NROWS,NCOLS,CV_32F);
	Mat top_wst_flat=Mat::zeros(hNROWS*nlayers,NCOLS,CV_16U);
	Mat bot_wst_flat=Mat::zeros(hNROWS*nlayers,NCOLS,CV_16U);
	vector<int> top_reg_neur(N_neur);
	vector<int> bot_reg_neur(N_neur);

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
		// initialize display matrix
		image3D[i]= Mat::zeros(NROWS,NCOLS,CV_32FC3);
	}
	
	cout<<"Initialized"<<endl;

	// Run the filter and calculate the maximum fluorescence value.
	Blur3D(top_image3D,top_gs,5,1);
	Blur3D(bot_image3D,bot_gs,5,1);
	ushort top_max=max3D(top_gs);
	ushort bot_max=max3D(bot_gs);
	ushort top_meanflu=mean3D(top_gs);
	ushort bot_meanflu=mean3D(bot_gs);
	ushort top_sd=var3D(top_gs,(float)top_meanflu);
	ushort bot_sd=var3D(bot_gs,(float)bot_meanflu);
	ushort top_orig_max=max3D(top_image3D);
	ushort bot_orig_max=max3D(bot_image3D);
	ushort top_minflu=top_meanflu+.5*top_sd;
	ushort bot_minflu=bot_meanflu+.5*bot_sd;

	cout<<"3D Gaussian filtering"<<endl;

	// Calculate MIP
	MIP(top_image3D,top_mip);
	MIP(bot_image3D,bot_mip);

	cout<<"MaxInt projections"<<endl;

    
    if(DOWS){
		// Calculate the watershed transform in 3D.
		cout<<"---------------------------------------"<<endl;
		cout<<"WATERSHED "<<endl;
		watershed3D(top_gs,top_wst,(ushort)top_max,top_minflu,tol,rad,MAX_ASSIGNMENT,0);
		watershed3D(bot_gs,bot_wst,(ushort)bot_max,bot_minflu,tol,rad,MAX_ASSIGNMENT,0);
		cout<<"---------------------------------------"<<endl;
		for(i=0;i<nlayers;i++) {
			top_wst[i].copyTo(top_wst_flat(Range(hNROWS*i,hNROWS*i+hNROWS),Range::all()));
			bot_wst[i].copyTo(bot_wst_flat(Range(hNROWS*i,hNROWS*i+hNROWS),Range::all()));
		}
		imwrite(daf7_WS.str(),top_wst_flat);
		imwrite(ins6_WS.str(),bot_wst_flat);
	} else {
		top_wst_flat = imread(daf7_WS.str(),IMREAD_UNCHANGED);
		bot_wst_flat = imread(ins6_WS.str(),IMREAD_UNCHANGED);
		for(i=0;i<nlayers;i++) {
			top_wst[i]=top_wst_flat(Range(hNROWS*i,hNROWS*i+hNROWS),Range::all());
			bot_wst[i]=bot_wst_flat(Range(hNROWS*i,hNROWS*i+hNROWS),Range::all());
		}
	}

    // Convert filtered image and WS transform to float and normalize. 
    for(i=0;i<nlayers;i++){
		top_gs[i].convertTo(top_gs_float[i],CV_32F,1./top_max);
		bot_gs[i].convertTo(bot_gs_float[i],CV_32F,1./bot_max);
		top_wst[i].convertTo(top_wst[i],CV_32F,1);
		bot_wst[i].convertTo(bot_wst[i],CV_32F,1);
		top_image3D[i].convertTo(top_image3D_float[i],CV_32F,1./top_max);
		bot_image3D[i].convertTo(bot_image3D_float[i],CV_32F,1./bot_max);
	}

	cout<<"Conversion to CV_32F"<<endl;
 
	// Convert & Normalize MIPs
	top_mip.convertTo(top_mip,CV_32F,1./top_orig_max);
	bot_mip.convertTo(bot_mip,CV_32F,1./bot_orig_max);

    // Create the window to display the 3D image and the WST
	cvNamedWindow( winname.str().c_str(), WINDOW_NORMAL | WINDOW_KEEPRATIO ); // Create a window for display.
	cvMoveWindow(winname.str().c_str(),0,0);
	resizeWindow( winname.str().c_str(), 2*NCOLS,2*NROWS);

    // Create a window for the MIP and show them
	cvNamedWindow( "Maximum Projection", WINDOW_NORMAL | WINDOW_KEEPRATIO ); // Create a window for display.
	cvMoveWindow("Maximum Projection",NCOLS*2,0);
	resizeWindow("Maximum Projection", NCOLS*3/2,NROWS*3/2);

	iomat param;
	param.stream = &logfile;
	float brightness=1;
	int brightness_MIP=7;
    
	// Define top and bottom matrices of single neurons
	vector<vector<Mat> > top_NeurVec(N_neur,vector<Mat>(nlayers));
	vector<vector<Mat> > bot_NeurVec(N_neur,vector<Mat>(nlayers));
	vector<Mat> top_NeurVec_MIP(N_neur);
	vector<Mat> bot_NeurVec_MIP(N_neur);
	ushort top_NeurTh[N_neur], bot_NeurTh[N_neur];

	// Define position vector
	vector<Point3i> top_NeurPos(N_neur), bot_NeurPos(N_neur); 
	// Define Mode vector
	vector<int> top_NeurMode(N_neur), bot_NeurMode(N_neur);

	// vectors of fluorescence voxels for each neuron 
	vector<vector<ushort> > top_NeurFl(N_neur);
	vector<vector<ushort> > bot_NeurFl(N_neur);
	for(k=0;k<N_neur;k++) {
		for(kp=0;kp<nlayers;kp++){
			top_NeurVec[k][kp] = Mat::zeros(hNROWS,NCOLS,CV_8U);
			bot_NeurVec[k][kp] = Mat::zeros(hNROWS,NCOLS,CV_8U);
		}
	}

    // This variable sets the neuron on which I am writing. 
	// It can be changed by clicking on the control panel.
	int curr_neur = 0;
	bool doanalyse=0;
    float top_neur_label[N_neur];
	for(i=0;i<N_neur;i++) top_neur_label[i]=0;
    float bot_neur_label[N_neur];
	for(i=0;i<N_neur;i++) bot_neur_label[i]=0;

    vector<Mat> top_wst_colored(nlayers);
    vector<Mat> bot_wst_colored(nlayers);
    for(k=0;k<nlayers;k++) {
        top_wst_colored[k] = Mat::zeros(hNROWS,NCOLS,CV_32FC3);
        bot_wst_colored[k] = Mat::zeros(hNROWS,NCOLS,CV_32FC3);
    }
    MakeColored(top_wst,top_wst_colored);
    MakeColored(bot_wst,bot_wst_colored);

    // Copy to image to display 
	for(i=0;i<nlayers;i++){
		top_image3D_float[i].copyTo(split[0]);
		top_image3D_float[i].copyTo(split[1]);
		top_image3D_float[i].copyTo(split[2]);
		merge(split,mer);
		mer.copyTo(image3D[i](Range(0,hNROWS),Range::all()));
		
		top_image3D_float[i].copyTo(split[2]);
		top_image3D_float[i].copyTo(split[1]);
		top_image3D_float[i].copyTo(split[0]);
		merge(split,mer);
        add(mer,0.4*top_wst_colored[i],mer);
		mer.copyTo(image3D[i](Range(hNROWS,NROWS),Range::all()));
	}

	// Copy MIPs to mip.
	top_mip.copyTo(mip(Range(0,hNROWS),Range::all()));
	bot_mip.copyTo(mip(Range(hNROWS,NROWS),Range::all()));
	// and display it.
	cvCreateTrackbar( "Brightness", "Maximum Projection", &brightness_MIP, 20,  NULL);

    // Set up of the GUI.
	int level=0;
	cvCreateTrackbar( "Layer", winname.str().c_str(), &level, 50,  NULL);
	imshow(winname.str().c_str(),brightness*image3D[level]);

    char c = '1';
	cout<<"DAF-7"<<endl;
	cout<<"Associate value to "<<neur_names[curr_neur]<<endl;
	logfile<<"DAF-7"<<endl; 
	logfile<<"Associate value to "<<neur_names[curr_neur]<<endl;
    while(c != 'q')	 {

		 if( c == ' ' ){
			 curr_neur=(curr_neur+1)%N_neur;
			 cout<<"Associate value to "<<neur_names[curr_neur]<<endl;
			 logfile<<"Associate value to "<<neur_names[curr_neur]<<endl;
		 }

		 if( c == 's' ) {
			 display_fun(top_NeurVec,curr_neur,top_gs);
		 }

		 if( c == 'x' ) reset_mask(top_NeurVec[curr_neur]);

		 imshow(winname.str().c_str(),image3D[level]);
	     imshow("Maximum Projection",10./brightness_MIP*mip);
		 param.state  = level;
		 param.reg_neur = &top_reg_neur[curr_neur];
		 param.input  = top_wst.begin();
		 param.label  = &top_neur_label[curr_neur];
		 param.output = top_NeurVec[curr_neur].begin();
	     cvSetMouseCallback(winname.str().c_str(),on_mouse,&param);
		 string disp = "Select "+neur_names[curr_neur];  
		 displayStatusBar(winname.str().c_str(), disp, 0 );
		 c=cvWaitKey(33);
	}

    // Copy to image to display 
	for(i=0;i<nlayers;i++){
		bot_image3D_float[i].copyTo(split[0]);
		bot_image3D_float[i].copyTo(split[1]);
		bot_image3D_float[i].copyTo(split[2]);
		merge(split,mer);
		mer.copyTo(image3D[i](Range(0,hNROWS),Range::all()));
		
		bot_image3D_float[i].copyTo(split[2]);
		bot_image3D_float[i].copyTo(split[1]);
		bot_image3D_float[i].copyTo(split[0]);
		merge(split,mer);
        add(mer,0.4*bot_wst_colored[i],mer);
        
		mer.copyTo(image3D[i](Range(hNROWS,NROWS),Range::all()));
	}

    c='1';
	curr_neur=0; // goes back to ASI-1
	cout<<"INS-6"<<endl;
	cout<<"Associate value to "<<neur_names[curr_neur]<<endl;
	logfile<<"INS-6"<<endl;
	logfile<<"Associate value to "<<neur_names[curr_neur]<<endl;
    while(c != 'q')	 {

		 if( c == ' ' ) {
			 curr_neur=(curr_neur+1)%N_neur;
			 cout<<"Associate value to "<<neur_names[curr_neur]<<endl;
			 logfile<<"Associate value to "<<neur_names[curr_neur]<<endl;
		 }

		 if( c == 's' ) {
			 display_fun(bot_NeurVec,curr_neur,bot_gs);
		 }

		 if( c == 'x' ) reset_mask(bot_NeurVec[curr_neur]);

		 imshow(winname.str().c_str(),image3D[level]);
	     imshow("Maximum Projection",10./brightness_MIP*mip);
		 param.state  = level;
		 param.reg_neur = &bot_reg_neur[curr_neur];
		 param.input  = bot_wst.begin();
		 param.label  = &bot_neur_label[curr_neur];
		 param.output = bot_NeurVec[curr_neur].begin();
	     cvSetMouseCallback(winname.str().c_str(),on_mouse,&param);
		 if(c == 'o' ) cout << "Value = "<<param.label << endl;
		 string disp = "Select "+neur_names[curr_neur];  
		 displayStatusBar(winname.str().c_str(), disp, 0 );
		 c=cvWaitKey(33);
	}

	cvDestroyAllWindows();
	// unlock
	ofstream unlock_file("lock");
	unlock_file<<0;
	unlock_file.close();

	ofstream top_outfile[N_neur];
	ofstream bot_outfile[N_neur];

	for(k=0;k<N_neur;k++){
		if(top_neur_label[k]!=0 || bot_neur_label[k]!=0) doanalyse=1;
	}

	if(!doanalyse){
		for(k=0;k<N_neur;k++){
			resfile<<neur_names[k]<<"0 0 [0, 0, 0] 0 0 0 0"<<endl;
			resfile<<neur_names[k]<<"0 0 [0, 0, 0] 0 0 0 0"<<endl;
		}
		return 0;
	}

	for(k=0;k<N_neur;k++){
		ostringstream top_namefile;
		ostringstream bot_namefile;
		top_namefile<<proc_folder<<'/'<<neur_names[k]<<"_ID"<<ID<<"_daf7dist_2.dat";
		bot_namefile<<proc_folder<<'/'<<neur_names[k]<<"_ID"<<ID<<"_ins6dist_2.dat";
		top_outfile[k].open(top_namefile.str().c_str());
		bot_outfile[k].open(bot_namefile.str().c_str());
	}
    
   cout<<"Processing daf-7"<<endl;

   // Store fluorescence values of each neuron in a vector and store it in an output file 
	for(k=0;k<nlayers;k++){
		for(kp=0;kp<N_neur;kp++) n_it[kp] = (top_NeurVec[kp][k]).begin<uchar>();
		for(it=top_gs[k].begin<ushort>();it!=top_gs[k].end<ushort>();++it){
			for(kp=0;kp<N_neur;kp++){
				if(*n_it[kp]!=0) {
					top_outfile[kp]<<*it<<endl;
					top_NeurFl[kp].push_back(*it);
				}
				n_it[kp]++;
			}
		}
	}
 
    // Sort the vector and calculate the threshold for each neuron
    for(k=0;k<N_neur;k++){
		if(top_NeurFl[k].size()>0){
			sort(top_NeurFl[k].begin(),top_NeurFl[k].end());
			top_NeurTh[k]=top_NeurFl[k][max(0,(int)(top_NeurFl[k].size()-top_reg_neur[k]*constants::neur_size[k]))];
		} else {
			top_NeurTh[k]=0;
		}
	}

    // Apply threshold to top_NeurVec 
	for(k=0;k<nlayers;k++){
		for(kp=0;kp<N_neur;kp++) n_it[kp] = (top_NeurVec[kp][k]).begin<uchar>();
		for(it=top_gs[k].begin<ushort>();it!=top_gs[k].end<ushort>();++it){
			for(kp=0;kp<N_neur;kp++){
				if(*n_it[kp]!=0 && *it<top_NeurTh[kp]) *n_it[kp]=0;
				n_it[kp]++;
			}
		}
	}

    // Calculate centers of neurons
    for(k=0;k<N_neur;k++) {
		Pt_tmp=Center(top_gs,top_NeurVec[k]);
		dilate(top_NeurVec[k][Pt_tmp.z],mask_tmp,element);
		mask_tmp=mask_tmp-top_NeurVec[k][Pt_tmp.z];
		mask_tmp.convertTo(mask_tmp_float,CV_32F);
		
		top_NeurPos[k]=Pt_tmp;
		top_NeurMode[k]=GetMode(top_gs[Pt_tmp.z],mask_tmp,top_neur_label[k]);
	}

   cout<<"Processing ins-6"<<endl;

   // Store fluorescence values of each neuron in a vector and store it in an output file 
   for(k=0;k<nlayers;k++){
		for(kp=0;kp<N_neur;kp++) n_it[kp] = (bot_NeurVec[kp][k]).begin<uchar>();
		for(it=bot_gs[k].begin<ushort>();it!=bot_gs[k].end<ushort>();++it){
			for(kp=0;kp<N_neur;kp++){
				if(*n_it[kp]!=0) {
					bot_outfile[kp]<<*it<<endl;
					bot_NeurFl[kp].push_back(*it);
				}
				n_it[kp]++;
			}
		}
	}
 
    // Sort the vector and calculate the threshold for each neuron
    for(k=0;k<N_neur;k++){
		if(bot_NeurFl[k].size()>0){
			sort(bot_NeurFl[k].begin(),bot_NeurFl[k].end());
			bot_NeurTh[k]=bot_NeurFl[k][max(0,(int)(bot_NeurFl[k].size()-bot_reg_neur[k]*constants::neur_size[k]))];
		} else {
			bot_NeurTh[k]=0;
		}
	}

    // Apply threshold to bot_NeurVec 
	for(k=0;k<nlayers;k++){
		for(kp=0;kp<N_neur;kp++) n_it[kp] = (bot_NeurVec[kp][k]).begin<uchar>();
		for(it=bot_gs[k].begin<ushort>();it!=bot_gs[k].end<ushort>();++it){
			for(kp=0;kp<N_neur;kp++){
				if(*n_it[kp]!=0 && *it<bot_NeurTh[kp]) *n_it[kp]=0;	
				n_it[kp]++;
			}
		}
	}

    // Calculate centers of neurons
    for(k=0;k<N_neur;k++) {
		Pt_tmp=Center(bot_gs,bot_NeurVec[k]);
		dilate(bot_NeurVec[k][Pt_tmp.z],mask_tmp,element);
		mask_tmp=mask_tmp-bot_NeurVec[k][Pt_tmp.z];
		mask_tmp.convertTo(mask_tmp_float,CV_32F);
		
		bot_NeurPos[k]=Pt_tmp;
		bot_NeurMode[k]=GetMode(bot_gs[Pt_tmp.z],mask_tmp,bot_neur_label[k]);
	}

	for(k=0;k<N_neur;k++){
		top_FL=FlCalc(top_NeurFl[k],top_NeurMode[k],k,top_reg_neur[k]);
		bot_FL=FlCalc(bot_NeurFl[k],bot_NeurMode[k],k,bot_reg_neur[k]);
		top_BG=BgCalc(top_NeurFl[k],top_NeurMode[k],k,top_reg_neur[k]);
		bot_BG=BgCalc(bot_NeurFl[k],bot_NeurMode[k],k,bot_reg_neur[k]);

		resfile<<neur_names[k]<<' '<<top_FL<<' '<<top_BG<<' '<<top_NeurPos[k]<<' '<<top_NeurMode[k]<<' '<<top_minflu<<' '<<top_NeurFl[k].size()<<' '<<top_reg_neur[k]<<endl;
		resfile<<neur_names[k]<<' '<<bot_FL<<' '<<bot_BG<<' '<<bot_NeurPos[k]<<' '<<bot_NeurMode[k]<<' '<<bot_minflu<<' '<<bot_NeurFl[k].size()<<' '<<bot_reg_neur[k]<<endl;
	}

    // SAVE NEURONS IMAGES
	for(k=0;k<N_neur;k++) top_NeurVec_MIP[k]=Mat::zeros(hNROWS,NCOLS,CV_8U);
	for(k=0;k<N_neur;k++) bot_NeurVec_MIP[k]=Mat::zeros(hNROWS,NCOLS,CV_8U);
	for(k=0;k<nlayers;k++){
		for(kp=0;kp<N_neur;kp++){
			top_NeurVec_MIP[kp] = top_NeurVec_MIP[kp]+top_NeurVec[kp][k];
			bot_NeurVec_MIP[kp] = bot_NeurVec_MIP[kp]+bot_NeurVec[kp][k];
		}
	}

    vector<Mat> splitMIP(3);
	Mat merMIP=Mat::zeros(hNROWS,NCOLS,CV_8UC3);

	splitMIP[1]=top_NeurVec_MIP[0];
	for(i=1;i<=1;i++) splitMIP[1]=splitMIP[1]+top_NeurVec_MIP[i];
	splitMIP[2]=bot_NeurVec_MIP[0];
	for(i=1;i<=3;i++) splitMIP[2]=splitMIP[2]+bot_NeurVec_MIP[i];
	splitMIP[0]=Mat::zeros(hNROWS,NCOLS,CV_8U);

	merge(splitMIP,merMIP);
    
	stringstream image_name;
	image_name<<proc_folder<<'/'<<"NS_ID"<<ID<<"_2.jpg";
	imwrite(image_name.str().c_str(),200*merMIP);
		
	return 0;

}

