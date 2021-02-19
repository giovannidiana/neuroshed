#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <fstream>
#include "include/handlers.h"
#include "include/iomat.h"
#include "include/flu_fun.h"
#include "include/constants.hpp"

using namespace std;
using namespace cv;
using namespace constants;

// Get Mode from a Matrix of swt*sht = 64*168 around the neuron in the focus plane
// which is assumed to correspond to the plane that cut the neuron at its center-of-fluorescence. 
int GetMode(const Mat &image_cut, const Mat &mask_cut,float val){
	vector<ushort> vec;
	MatConstIterator_<ushort> image_cut_it;
	MatConstIterator_<uchar> mask_cut_it=mask_cut.begin<uchar>();
	vector<ushort>::iterator it;
	vector<ushort>::const_iterator it_const;
	vector<int> counts;
	vector<ushort> diff;
	int maxfl=0;
	int curr=-1;
	int ind=0;
	int i;
	int modeval;
    for(image_cut_it=image_cut.begin<ushort>();image_cut_it!=image_cut.end<ushort>();++image_cut_it){
		if(*mask_cut_it!=0){
			vec.push_back(*image_cut_it);
		}
		++mask_cut_it;
	}
    
	sort(vec.begin(),vec.end());
	for(it=vec.begin();it!=vec.end();++it){
		if(*it!=curr){
			counts.push_back(1);
			diff.push_back(*it);
			ind++;
			curr=*it;
		} else {
			counts[ind-1]++;
		}
	}

	for(i=0;i<ind;i++){
		if(counts[i]>maxfl){
			maxfl=counts[i];
			modeval=diff[i];
		}
	}

	return modeval;
}


int FlCalc(const vector<ushort>& vec,int mode,int neur_id,int mul){
	vector<ushort> vec_sort(vec);
	reverse(vec_sort.begin(),vec_sort.end());
	vector<ushort>::iterator it;

	int FL=0;
	int counter=0;
   
	for(it=vec_sort.begin();it!=vec_sort.end();++it){
		counter++;
		if(counter<mul*neur_size[neur_id]) FL+=*it-mode;
	}

	return FL;

}

int BgCalc(const vector<ushort>& vec,int mode,int neur_id,int mul){
	//vector<ushort> vec_sort(vec);
	//sort(vec_sort.begin(),vec_sort.end());
	//reverse(vec_sort.begin(),vec_sort.end());
	//vector<ushort>::iterator it;

	int BG=0;
	//int counter=0;
   
	//for(it=vec_sort.begin();it!=vec_sort.end();++it){
	//	counter++;
	//	if(counter<neur_size) FL+=*it-mode;
	//}

	BG=std::min((int)vec.size(),mul*neur_size[neur_id])*mode;

	return BG;

}



