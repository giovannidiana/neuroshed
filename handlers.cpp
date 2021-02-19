#include "opencv2/highgui.hpp"
#include "opencv2/core/opengl.hpp"
#include <iostream>
#include "include/handlers.h"
#include "include/iomat.h"
#include <string>
#include "include/constants.hpp"
#include "include/GL.hpp"

using namespace std;
using namespace cv;
using namespace constants;

void on_mouse(int event, int x, int y, int flags, void* param){
	if  ( event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN)	{
		iomat* IOM = (iomat*)param;
		int state  = IOM->state;
		int k=0;
		int regneur=1;
		int dilation_size=1;
		int xp=x,yp=max(0,y-hNROWS);
		vector<Mat>::iterator in = IOM->input;
		vector<Mat>::iterator ou = IOM->output;
		Mat dil_tmp;  
		MatIterator_<float> it;
		MatIterator_<uchar> it_out;

		Mat element = getStructuringElement( MORPH_ELLIPSE, 
		                   Size( 2*dilation_size + 1, 2*dilation_size+1 ),
						   Point( dilation_size, dilation_size ) 
					  );
		
		float value = (*(in+state)).at<float>(yp,xp);
		if(event == EVENT_LBUTTONDOWN){
			cout << "Value = "<<value <<" pos = ["<<x<<","<<y<<','<<state<<"]"<<endl;
			*(IOM->stream) <<value <<' '<<x<<" "<<y<<" "<<state<<' '<<1<<endl;
			*(IOM->reg_neur) = 1;
		} else if(event == EVENT_RBUTTONDOWN){
			cout<<"Enter number of neurons in the region and press ENTER"<<endl;
			cin>>regneur;
			cout <<"Value = "<<value <<" pos = ["<<x<<","<<y<<','<<state<<"]"<<' '<<regneur<<endl;
			*(IOM->stream) <<value<<' ' <<x<<" "<<y<<" "<<state<<' '<<regneur<<endl;
			*(IOM->reg_neur) = regneur;
		}

		*(IOM->label)=value;

			    
		if(value!=0){
			for(k=0;k<constants::nlayers;k++){
				it_out = (*(ou+k)).begin<uchar>();
				for(it=(*(in+k)).begin<float>();it!=(*(in+k)).end<float>();++it){
					if(*it==value) *it_out = 1;
					++it_out;
				}
			}

			for(k=0;k<constants::nlayers;k++){
				dilate(*(ou+k), dil_tmp, element );
				*(ou+k) = dil_tmp.clone();
			}
			
		}

	}
}

void on_mouse_disp(int event, int x, int y, int flags, void* param){
	Point2i* Pt = (Point2i*)param;
	Pt->x=x;
	Pt->y=y;
}

void display_fun_2D(const vector<Mat>& mask,int neur_id,const vector<Mat>& image){
    vector<Mat> output(image.size());
	vector<ushort> vec;
	MatConstIterator_<ushort> image_it;
	MatConstIterator_<uchar>  mask_it;
	MatIterator_<ushort> it;
	string neuron=constants::neur_names[neur_id];

	int k=0;

	for(k=0;k<image.size();k++){
		mask_it=mask[k].begin<uchar>();
		for(image_it=image[k].begin<ushort>();image_it!=image[k].end<ushort>();++image_it){
			if(*mask_it!=0) vec.push_back(*image_it);
			++mask_it;
		}
	}

    if(vec.size()==0) return ;

	sort(vec.begin(),vec.end());
	ushort th=vec[max(0,(int)(vec.size()-constants::neur_size[neur_id]))];
	
	for(k=0;k<image.size();k++){
		output[k] = Mat::zeros(hNROWS,NCOLS,CV_16U);
		image[k].copyTo(output[k],mask[k]);
		for(it=output[k].begin<ushort>();it!=output[k].end<ushort>();++it){
			if(*it>th) {
				*it=65535;
			} else *it = 0;
		}
	}

    // Create the window to display the 3D image and the WST
	cvNamedWindow(neuron.c_str(), WINDOW_NORMAL | WINDOW_KEEPRATIO ); // Create a window for display.
	resizeWindow( neuron.c_str(), NCOLS,NROWS);
	moveWindow(neuron.c_str(),NCOLS*2,0);

    // Set up of the GUI.
	int level=0;
	cvCreateTrackbar( "Layer", neuron.c_str(), &level, 50,  0);

    while(cvWaitKey(30) != 'q')	{ 
		imshow(neuron.c_str(),output[level]);
	}

	destroyWindow(neuron);

}

void display_fun(const vector<vector<Mat> >& mask,int neur_id,const vector<Mat>& image){
	int N_neur=constants::N_neur;
    vector<vector<float> > output(N_neur);
	float center[3]={0,0,0};
	vector<vector<ushort> > vec(N_neur);
	MatConstIterator_<ushort> image_it;
	vector<MatConstIterator_<uchar> >  mask_it(N_neur);
	MatIterator_<ushort> it;
	string neuron=constants::neur_names[neur_id];

	int k=0,kn;
	int ip,jp;
	char ** a;

	for(k=0;k<image.size();k++){
		for(kn=0;kn<N_neur;kn++) mask_it[kn]=mask[kn][k].begin<uchar>();
		for(image_it=image[k].begin<ushort>();image_it!=image[k].end<ushort>();++image_it){
			for(kn=0;kn<N_neur;kn++){
				if(*mask_it[kn]!=0) vec[kn].push_back(*image_it);
				++mask_it[kn];
			}
		}
	}

	vector<ushort> th(N_neur);
	for(kn=0;kn<N_neur;kn++) {
		if(vec[kn].size()!=0){
			sort(vec[kn].begin(),vec[kn].end());
			th[kn]=vec[kn][max(0,(int)(vec[kn].size()-constants::neur_size[kn]))];
		} else {
			th[kn]=-1;
		}
	}

    bool cont=0;
	for(kn=0;kn<N_neur;kn++){
		if(th[kn]!=-1) {
			cont=1;
			break;
		}
	}
    
	if(!cont) return;
    int counter=0;
	for(k=0;k<image.size();k++){
		for(ip=0;ip<hNROWS;ip++){
		for(jp=0;jp<NCOLS;jp++){
			for(kn=0;kn<N_neur;kn++){
				if(mask[kn][k].at<uchar>(ip,jp)!=0 && image[k].at<ushort>(ip,jp)>th[kn]) {
					output[kn].push_back(jp);
					output[kn].push_back(265-ip);
					output[kn].push_back(k);
					counter++;
					break;
				}
			}
		}}
	}

    for(kn=0;kn<N_neur;kn++){
		for(k=0;k<output[kn].size()/3;k++){
			center[0]+=1.0*output[kn][3*k]/counter;
			center[1]+=1.0*output[kn][3*k+1]/counter;
			center[2]+=1.0*output[kn][3*k+2]/counter;
	    }
	}
	cout<<center[0]<<' '<<center[1]<<' '<<center[2]<<endl;

    for(kn=0;kn<N_neur;kn++){
		for(k=0;k<output[kn].size()/3;k++){
			output[kn][3*k]+=-center[0];
			output[kn][3*k+1]+=-center[1];
			output[kn][3*k+2]+=-center[2];
		}
	}

    void* output_ptr = &output; 

	cvNamedWindow( "3D", WINDOW_OPENGL ); // Create a window for display.
	resizeWindow("3D",NCOLS*3/2,hNROWS*3/2);
	moveWindow("3D",NCOLS,hNROWS);
	setOpenGlContext("3D");
	setOpenGlDrawCallback("3D",on_GL,output_ptr);
	char c='0';
	while(c!='q'){
		c=cvWaitKey(33);
		updateWindow("3D");
        cvSetMouseCallback("3D",on_mouse_GL,0);
	}
	destroyWindow("3D");
	
}

void reset_mask(vector<Mat> & mask){
	
	int k;
	MatIterator_<uchar> it;

    for(k=0;k<mask.size();k++){
		for(it=mask[k].begin<uchar>();it!=mask[k].end<uchar>();++it){
			*it=0;
		}
	}
	

}


