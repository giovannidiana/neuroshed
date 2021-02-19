#include <opencv2/highgui.hpp>
#include <opencv2/core/opengl.hpp>
#ifdef __APPLE__
     #include<OpenGL/gl.h>
#else
     #include<GL/gl.h>
#endif
#include<GL/freeglut.h>
#include <iostream>
#include "include/constants.hpp"
#include <armadillo>
#include"include/GL.hpp"

using namespace std;
using namespace arma;
using namespace constants;

static vector<double> angle(1);
static vector<double> rot(3);

int W=NCOLS;
int H=hNROWS;
static float x0r=0;
static float y0r=0;
static double x1r=0;
static double y1r=0;
static double z1r=0;
static int rbdown=0;
static int lbdown=0;
static float sc=1;

static double pi=acos(-1);

float tx0=0,ty0=0;
float tx=0,ty=0;
float a1=0,a2=0;

mat RM3D(double phi, double x,double y,double z){
    mat R(3,3);
	R(0,0)=cos(phi)+pow(x,2)*(1-cos(phi));
	R(0,1)=x*y*(1-cos(phi))-z*sin(phi);
	R(0,2)=x*z*(1-cos(phi))+y*sin(phi);
	R(1,0)=y*x*(1-cos(phi))+z*sin(phi);
	R(1,1)=cos(phi)+pow(y,2)*(1-cos(phi));
	R(1,2)=y*z*(1-cos(phi))-x*sin(phi);
	R(2,0)=z*x*(1-cos(phi))-y*sin(phi);
	R(2,1)=z*y*(1-cos(phi))+x*sin(phi);
	R(2,2)=cos(phi)+pow(z,2)*(1-cos(phi));

	return R;
}

static mat GlobalR=RM3D(1*pi/180,1,0,0);

void on_mouse_GL(int event,int x,int y, int flags, void* param){
	float normv;
	vec rvec(3);
/*
    if(cvWaitKey(33)=='l'){
		angle.push_back(10);
		rot.push_back(0);
		rot.push_back(1);
		rot.push_back(0);
		GlobalR=GlobalR*RM3D(angle.back()/180.*pi,
		                     rot[3*(angle.size()-1)],
		                     rot[3*(angle.size()-1)+1],
		                     rot[3*(angle.size()-1)+2]);
    }
*/
	if(event == cv::EVENT_LBUTTONDOWN){
		tx0=x,ty0=y;
		lbdown=1;
        rot.resize(3);
        rot[0]=1;rot[1]=rot[2]=0;
        angle.resize(1);
        angle[0]=1;
        GlobalR=RM3D(1*pi/180,1,0,0);
	}
	if(event == cv::EVENT_LBUTTONUP){
		tx=tx+x-tx0;ty=ty+ty0-y;
		lbdown=0;
	}

	if(event == cv::EVENT_RBUTTONDOWN){
	    x0r=x;
		y0r=y;
		angle.push_back(0);
		rot.push_back(1);
		rot.push_back(0);
		rot.push_back(0);
		rbdown=1;
	}
	if(event == cv::EVENT_RBUTTONUP){
	    rbdown=0;
		GlobalR=GlobalR*RM3D(angle.back()/180*pi,
		                     rot[3*(angle.size()-1)],
		                     rot[3*(angle.size()-1)+1],
		                     rot[3*(angle.size()-1)+2]);
    }

	if(event == cv::EVENT_MOUSEMOVE){
	    
		//if(lbdown==1) glViewport(tx+x-tx0,ty+ty0-y,W,H);
		
		if(rbdown==1){
			rvec[0]=-(y)+(y0r);
			rvec[1]=-x+x0r;
			rvec[2]=0;
			normv=norm(rvec);
			
			rvec=inv(GlobalR)*rvec;
			
			x1r=rvec[0]/normv;
			y1r=rvec[1]/normv;
			z1r=rvec[2]/normv;
			
			angle.back() = normv;
			rot[3*(angle.size()-1)]=x1r;
			rot[3*(angle.size()-1)+1]=y1r;
			rot[3*(angle.size()-1)+2]=z1r;
		}
	}
}
	


void on_GL(void* param)
{
    int i,j;
	angle[0]=1;
	rot[0]=1;rot[1]=0;rot[2]=0;
	float zspacing=3.;
	vector<vector<float> > pts=*((vector<vector<float> >*)param);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable     (GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glOrtho(-336,336,-128,128,-zspacing*100,zspacing*100);

	for(i =0 ; i<angle.size();i++){
	     glRotated(angle[i],rot[3*i],rot[3*i+1],rot[3*i+2]);
	}

    glColor4ub(128,128,128,100);
    glBegin(GL_LINES);
      glVertex3f(-336,0,0);
      glVertex3f(336,0,0);
      glVertex3f(0,0,-zspacing*100);
      glVertex3f(0,0,zspacing*100);
/*      for(i=0;i<15;i++){
          for(j=0;j<15;j++){
              glVertex3f(0,-128+hNROWS/15*i,-zspacing*100);
              glVertex3f(0,-128+hNROWS/15*i, zspacing*100);
              glVertex3f(0,-128, -zspacing*100+zspacing*200/15*i);
              glVertex3f(0,128, -zspacing*100+zspacing*200/15*i);
          }
      }
      */
    glEnd();
    
    glColor4ub(100,100,100,100);
    glBegin(GL_QUADS);
       glVertex3f(0,-128,zspacing*100);
       glVertex3f(0,128,zspacing*100);
       glVertex3f(0,128,-zspacing*100);
       glVertex3f(0,-128,-zspacing*100);
    glEnd();

    glTranslated(-336,0,0);
    glutSolidSphere( 5.0, 20.0, 20.0); 
    glTranslated(336,0,0);
	
    glTranslated(0,0,-zspacing*100);
    glutSolidSphere( 5.0, 20.0, 20.0); 
    glTranslated(0,0,zspacing*100);

    glBegin(GL_LINES);
	for(int kn=0;kn<constants::N_neur;kn++){
		if(pts[kn].size()>0){
			for (i = 0; i < pts[kn].size()/3; i++) {
			    glColor4ub(255-255/constants::N_neur*kn,255/constants::N_neur*kn,0,100);
				glVertex3f(pts[kn][3*i],pts[kn][3*i+1],zspacing*pts[kn][3*i+2]);
				glVertex3f(pts[kn][3*i],pts[kn][3*i+1],zspacing*pts[kn][3*i+2]+zspacing);
			}
		}
	}
	glEnd();
}
