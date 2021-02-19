/* 
09/03/2017: CODE001. Bug corrected.
            CODE002. Changed the else condition. Before it was only checked that the size of ass_tuple was larger than one but this is always true. 
                     It has been replaced with the constraint ass_tuple[0][0]!=0 which guarantees that the current point has at least one non zero
					 nearest association.
*/

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include "include/handlers.h"
#include "include/iomat.h"
#include <algorithm>
#include <iomanip>
#include<boost/tuple/tuple.hpp>
#include<boost/array.hpp>
#include<boost/tuple/tuple_io.hpp>

using namespace std;
using namespace cv;

struct SortArray{
    inline bool operator() (const boost::array<int,2>& t1, const boost::array<int,2>& t2){
        
        return (t1[1] < t2[1]);
    }
};

void MakeColored(const vector<Mat> &image3D, vector<Mat> &image3D_out){
    MatConstIterator_<float> it;
    MatIterator_<Vec3f> it3;
    int layers=image3D.size();
    int i,k;
    Vec3f tmpcol;

    for(k=0;k<layers;k++){
        it3=image3D_out[k].begin<Vec3f>();
        for(it=image3D[k].begin<float>();it!=image3D[k].end<float>();++it){
            if(*it!=0) {
                tmpcol[0]=fmod((*it)*sqrt(2), 1);
                tmpcol[1]=fmod((*it)*sqrt(3), 1);
                tmpcol[2]=fmod((*it)*sqrt(5), 1);
                *it3=tmpcol;
            }
            ++it3;
        }
    }
}      

ushort max3D(const vector<Mat> &image){
    ushort maxi=0;
	MatConstIterator_<ushort> it,end;
	int k;
	int layers=image.size();
	assert(image[0].type()==CV_16U);

	for(k=0;k<layers;k++){
		for(it=image[k].begin<ushort>(),end=image[k].end<ushort>();it!=end;++it){
		   if(*it>maxi) maxi=*it;
		}
	}
	return maxi;
}

float mean3D(const vector<Mat> &image){
    float mean=0;
	MatConstIterator_<ushort> it;
	int k;
	int layers=image.size();
	assert(image[0].type()==CV_16U);

	for(k=0;k<layers;k++){
		for(it=image[k].begin<ushort>();it!=image[k].end<ushort>();++it){
		   mean+=*it;
		}
	}
	return mean/(image[0].cols*image[0].rows*layers);
}

float var3D(const vector<Mat> &image,float mean){
    float var=0;
	MatConstIterator_<ushort> it;
	int k;
	int layers=image.size();
	assert(image[0].type()==CV_16U);

	for(k=0;k<layers;k++){
		for(it=image[k].begin<ushort>();it!=image[k].end<ushort>();++it){
		   var+=pow(*it-mean,2);
		}
	}
	return sqrt(var/(image[0].cols*image[0].rows*layers));
}

void Blur3D(const vector<Mat> &image, vector<Mat> &out_image, int rad,int blurz){
	int layers=image.size();
	int i,k,kp;
	float sum;
	vector<Mat> gbXY(layers);
	for(i=0; i<layers; i++){
		GaussianBlur(image[i],gbXY[i],Size(rad,rad),rad,rad);
	}
	vector<MatIterator_<ushort> > it(gbXY.size());
	vector<MatIterator_<ushort> > it_out(gbXY.size());

	for(k=0;k<layers;k++) {
		it[k]=gbXY[k].begin<ushort>();
		it_out[k] = out_image[k].begin<ushort>();
	}

	while(it[0]!=gbXY[0].end<ushort>()){
		for(k=0;k<layers;k++){
			sum=0;
			for(kp=max(0,k-blurz);kp<min(layers,k+blurz+1);kp++)
				sum+=*it[kp];

			*it_out[k]=(ushort)(sum/(2*blurz+1));
		}
		
		for(k=0;k<layers;k++){
			it[k]++;
			it_out[k]++;
		}
	}
}

void watershed3D(const vector<Mat> &image, vector<Mat> &wst, ushort maxflu,ushort minflu,int tol,int rad,int MAX_ASSIGNMENT,bool verbose){

    // The idea is to start from the maximum intensity voxels and 
	// propagate the flooding.
	int i,j,k;
	int iloc;
	int kp;
	int r,c,rloc,cloc;
	int ncol=image[0].cols;
	int nrow=image[0].rows;
	int nz=image.size();
	int f;
    int kmeans_n=3;
    bool KMEANS=false;
    int npoints,counter;
	ushort tmp,ass_tmp,ass_nei;
	Mat mat_tmp(rad,rad,CV_16UC1,Scalar(0));
	ushort* data;
	ushort* out_data;
	boost::array<int,3> P_temp;
	int assign=0;
	bool sw;
	MatIterator_<ushort> it,end;
    MatIterator_<ushort> it_wst;
    MatConstIterator_<ushort> it_const;
    
    vector<double> tmp_dist;
    vector<boost::array<int,3> > peaks;
    vector< boost::array<int,2> > ass_tuple;
    int ass_tuple_tot=0;

    boost::array<int,2> tmp_tup;
    
	// We scan from max to the minimum intensity.
	f=maxflu+tol;
	cout<<minflu<<' '<<maxflu<<endl;

	while(f>minflu /* && assign<MAX_ASSIGNMENT */){
		f+=-tol;
		if(verbose){
			cout<<"Scan fluorescence "<<setw(4)<<f
			    <<" assignment "<<setw(5)<<assign<<"     \r"<<flush;
		}
	    for(k=0;k<nz;k++){
	        data=(ushort*)image[k].data;
	        out_data=(ushort*)wst[k].data;

	        for(i=0;i<ncol*nrow;i++){
				tmp=*data;
				//if(i==100601) cout<<tmp<<endl;
				if(tmp<=f && tmp>f-tol && *out_data==0){
					r=i/ncol;
					c=i%ncol;
					
                    ass_tuple.resize(1);
                    ass_tuple[0][0]=0;
                    ass_tuple[0][1]=0;

					for(kp=max(0,k-2);kp<min(nz,k+3);kp++){
						mat_tmp=wst[kp](Range(max(0,r-rad),min(nrow,r+rad+1)),Range(max(0,c-rad),min(ncol,c+rad+1)));
						iloc=0;
						for(it=mat_tmp.begin<ushort>(),end=mat_tmp.end<ushort>();it!=end;++it){
							rloc=iloc/(2*rad+1)-rad;
							cloc=iloc%(2*rad+1)-rad;
                            sw=0;
                            if(true){
								ass_nei=*it;
								if(ass_nei!=0) {
                                    if(ass_tuple[0][0]==0){
                                        ass_tuple[0][0]=ass_nei;
                                        ass_tuple[0][1]=1;
                                    } else {
                                        for(j=0;j<ass_tuple.size();j++){
                                            if(ass_nei==ass_tuple[j][0]){
                                                ass_tuple[j][1]++;
                                                sw=1;
                                                break;
                                            }
                                        }

                                        if(sw==0){
                                            tmp_tup[0]=ass_nei;
                                            tmp_tup[1]=1;
                                            ass_tuple.push_back(tmp_tup);
                                        }
                                    }
                                }
                            }
                            
                            iloc++;
                        }
                    }
					

                    if(ass_tuple[0][0]==0 && *data>1.5*minflu){
                        assign++;
                        P_temp[0]=r; P_temp[1]=c; P_temp[2]=k;
                        peaks.push_back(P_temp);
                        *out_data=assign;
                    } else if(ass_tuple[0][0]!=0) { // CODE002
                        sort(ass_tuple.begin(),ass_tuple.end(),SortArray());
                        ass_tuple_tot=0;
                        tmp_dist.resize(0);
                        for(j=0;j<ass_tuple.size();++j) {
                            ass_tuple_tot+=ass_tuple[j][1]; // CODE001
                            tmp_dist.push_back(sqrt(pow(r-peaks[ass_tuple[j][0]-1][0],2)+
                                                    pow(c-peaks[ass_tuple[j][0]-1][1],2)+
                                                    6.2*pow(k-peaks[ass_tuple[j][0]-1][2],2)));
                        }

                        // Use the nearest association
                        if(*min_element(tmp_dist.begin(),tmp_dist.end())<20){
                            *out_data=ass_tuple[ distance(tmp_dist.begin(),min_element(tmp_dist.begin(),tmp_dist.end()))][0];
                        }
                        
                        // Use the most associated index
                        /*
                        tmp_tup = *max_element(ass_tuple.begin(),ass_tuple.end(),SortArray());
                        if(1.0*tmp_tup[1]/ass_tuple_tot>0.6 &&
                           sqrt(pow(r-peaks[tmp_tup[0]-1][0],2)+
                                pow(c-peaks[tmp_tup[0]-1][1],2)+
                                pow(k-peaks[tmp_tup[0]-1][2],2))<20){
                            *out_data=tmp_tup[0];
                        }
                        */
                    }   

                }// close if(tmp==f)
				
				// increase pointers
				data++;
				out_data++;

				
			} // close for on slice
		} // close for on stack
	} // close flooding

    npoints=assign;
    
    if(KMEANS){
    for(i=1;i<=npoints;i++){
        vector<float> tmp_vec;
        for(k=0;k<nz;k++){
            it_const=image[k].begin<ushort>();
            for(it_wst=wst[k].begin<ushort>(); it_wst!=wst[k].end<ushort>();++it_wst){
                if(*it_wst==i) tmp_vec.push_back(*it_const);
                ++it_const;
            }
        }
        if(verbose) cout<<i<<'\r'<<flush;

        if(tmp_vec.size()>kmeans_n){
            Mat tmp_mat(tmp_vec.size(),1,CV_32F,&tmp_vec[0]);
            Mat labels;
            Mat centers;
            int uselab;

            kmeans(tmp_mat,kmeans_n,labels,
                   TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 2, .1),
                   10,KMEANS_PP_CENTERS,centers);

            float min_val=centers.at<float>(0,0);
            float max_val=centers.at<float>(0,0);
            int min_ind=0;
            int max_ind=0;
            for(j=0;j<kmeans_n;j++) {
                if(min_val>centers.at<float>(j,0)) {
                    min_val=centers.at<float>(j,0);
                    min_ind=j;
                }
                if(max_val<centers.at<float>(j,0)) {
                    max_val=centers.at<float>(j,0);
                    max_ind=j;
                }
            }

            counter=0;
            for(k=0;k<nz;k++){
                for(it_wst=wst[k].begin<ushort>(); it_wst!=wst[k].end<ushort>();++it_wst){
                    if(*it_wst==i){
                        if(labels.at<int>(counter)==min_ind) *it_wst=0;
                        counter++;
                    }
                }
            }
        } else {
            for(k=0;k<nz;k++){
                for(it_wst=wst[k].begin<ushort>(); it_wst!=wst[k].end<ushort>();++it_wst){
                    if(*it_wst==i) *it_wst=0;
                }
            }
        }

    }
    } // end if
    // 4th step: select by size:

    if(true){
    for(i=1;i<=npoints;i++){
        counter=0;
        for(k=0;k<nz;k++){
            it_const=image[k].begin<ushort>();
            for(it_wst=wst[k].begin<ushort>(); it_wst!=wst[k].end<ushort>();++it_wst){
                if(*it_wst==i) counter+=*it_const;
            }
        }

        if(counter<10e4){
            for(k=0;k<nz;k++){
                for(it_wst=wst[k].begin<ushort>(); it_wst!=wst[k].end<ushort>();++it_wst){
                    if(*it_wst==i) *it_wst=0;
                }
            }
        }
        //cout<<i<<'\r'<<flush;
        if(verbose) cout<<i<<' '<<counter<<endl;
    }

    cout<<endl;
    }



}

void MIP(const vector<Mat>& image, Mat &mp){
	int nz=image.size();
	int i,k;
	MatIterator_<ushort> it;
	vector<MatConstIterator_<ushort> > it_const(nz);
	for(i=0;i<nz;i++) it_const[i]=image[i].begin<ushort>();
	for(it=mp.begin<ushort>();it!=mp.end<ushort>();it++){
		*it=0;
		for(k=0;k<nz;k++) *it=max(*it,*it_const[k]);
		for(k=0;k<nz;k++) it_const[k]++;
	}
}

Point3f Center(const vector<Mat> &image,const vector<Mat> &mask){
	MatConstIterator_<ushort> it;
	MatConstIterator_<uchar> it_mask;
	int col=image[0].cols;
	int nz=image.size();
	int x,y,z;
	int i=0;
	Point3f cen(0,0,0);
	float w=0;

	for(int k=0;k<nz;k++){
		it=image[k].begin<ushort>();
		i=0;
		for(it_mask=mask[k].begin<uchar>();it_mask!=mask[k].end<uchar>();it_mask++){
			if(*it_mask!=0){
				y=i%col;
				x=i/col;
				z=k;
				w+=*it;
				cen.x+=x*(*it); cen.y+=y*(*it); cen.z+=z*(*it);
			}
			i++;
			it++;
		}
	}
	if(w>0) cen=cen*1.0/w;

	return cen;
}

