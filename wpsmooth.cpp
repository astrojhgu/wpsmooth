#include <cstdlib>
#include <cassert>
#include "spline.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <vector>
using namespace std;

struct rparams
{
  vector<double> w_list;
  double lbd;
  double sf;
  int N_grid;
  vector<double> xn;
  vector<double> yn;
  vector<double> Xn;
  vector<int> mn;
};

double pw(double x,void* params)
{
  const rparams& rp=(*static_cast<rparams*>(params));
  const vector<double>& w_list=rp.w_list;
  double lbd=rp.lbd;
  double sf=rp.sf;
  double N_grid=rp.N_grid;
  const vector<double>& xn=rp.xn;
  const vector<double>& yn=rp.yn;
  const vector<double>& Xn=rp.Xn;
  const vector<int>& mn=rp.mn;
  double result=sf;
  for(int i=0;i<w_list.size();++i)
    {
      result*=(x-w_list[i]);
    }
  return result;
}

double psi(double x)
{
  return x;
}

int func (const gsl_vector * Y, void *params,
		  gsl_vector * f)
{
  static int cnt=0;
  const rparams& rp=(*static_cast<rparams*>(params));
  const vector<double>& w_list=rp.w_list;
  double lbd=rp.lbd;
  double sf=rp.sf;
  int N_grid=rp.N_grid;
  const vector<double>& xn=rp.xn;
  const vector<double>& yn=rp.yn;
  const vector<double>& Xn=rp.Xn;
  const vector<int>& mn=rp.mn;
  vector<double> result(2*N_grid);
  //result.reserve(2*N_grid);
  

#pragma omp parallel for schedule(dynamic)
  for(int j=1;j<N_grid-1;++j)
    {
      double Yj1=gsl_vector_get(Y,j+1);
      double Yj_1=gsl_vector_get(Y,j-1);
      double Yj=gsl_vector_get(Y,j);
      double hj1=gsl_vector_get(Y,N_grid+j+1);
      double hj=gsl_vector_get(Y,N_grid+j);
      double hj_1=gsl_vector_get(Y,N_grid+j-1);
      double Delta_j=((Xn[j+1]-Xn[j])*(Xn[j]-Xn[j-1]));
      result[2*(j-1)]=((Yj1-2*Yj+Yj_1)-pw(Xn[j],params)*exp(hj)*Delta_j);
      double s=0;
      for(int i=0;i<mn.size();++i)
	{
	  s+=max(Xn.at(j)-xn.at(i),0.)*psi(yn.at(i)-gsl_vector_get(Y,mn.at(i)));
	}
      s/=(-2*lbd);
      result[2*(j-1)+1]=((hj1-2*hj+hj_1)-pw(Xn[j],params)*exp(hj)*Delta_j*s);
    }
  result[result.size()-4]=(gsl_vector_get(Y,N_grid)-gsl_vector_get(Y,N_grid+1));
  result[result.size()-3]=(gsl_vector_get(Y,2*N_grid-1)-gsl_vector_get(Y,2*N_grid-2));
  double s1=0;
  double s2=0;
  ofstream ofs("result.dat");
  for(int i=0;i<mn.size();++i)
    {
      ofs<<xn[i]<<"\t"<<yn[i]<<"\t"<<gsl_vector_get(Y,mn[i])<<endl;
      s1+=psi(yn[i]-gsl_vector_get(Y,mn[i]));
      s2+=xn[i]*psi(yn[i]-gsl_vector_get(Y,mn[i]));
    }
  result[result.size()-2]=(s1);
  result[result.size()-1]=(s2);
  double resid=0;
  assert(result.size()==f->size);
  for(int i=0;i<result.size();++i)
    {
      //      cout<<result[i]*result[i]<<" ";
      resid+=result[i]*result[i];
      gsl_vector_set(f,i,result[i]);
    }
  resid/=result.size();
  resid=sqrt(resid);
  if(cnt++%1==0)
    {
      cout<<cnt<<"\t"<<resid<<endl;
    }
  return GSL_SUCCESS;
}

int print_state (size_t iter, gsl_multiroot_fsolver * s)
{
  cout<<iter<<endl;
}


int main (void)
{
  rparams p;
  p.lbd=1;
  p.sf=1;
  ifstream ifs("total_signal.qdp");
  spline<double> spl;
  for(;;)
    {
      double x,y;
      ifs>>x>>y;
      if(!ifs.good())
	{
	  break;
	}
      spl.push_point(log(x),log(y));
      p.xn.push_back(x);
      p.yn.push_back(y);
    }
  spl.gen_spline(0,0);
  p.Xn.push_back(p.xn[0]);
  p.mn.push_back(0);

  for(int i=1;i<p.xn.size();++i)
    {
      int n=2;
      for(int j=0;j<n-1;++j)
	{
	  p.Xn.push_back(p.xn[i]+(p.xn[i]-p.xn[i-1])/n*(j+1));
	}
      p.mn.push_back(p.Xn.size());
      p.Xn.push_back(p.xn[i]);
    }

  p.N_grid=p.Xn.size();
  
  const gsl_multiroot_fsolver_type *T;
  gsl_multiroot_fsolver *s;
  
  int status;
  size_t i, iter = 0;
  
  const size_t n = p.N_grid*2;

  gsl_multiroot_function f = {&func, n, &p};
  
  //double x_init[2] = {-10.0, -5.0};

  gsl_vector *x = gsl_vector_alloc (p.N_grid*2);
  
  //gsl_vector_set (x, 0, x_init[0]);
  //gsl_vector_set (x, 1, x_init[1]);
  
  for(int i=0;i<p.Xn.size();++i)
    {
      gsl_vector_set(x,i,exp(spl.get_value(log(p.Xn[i]))));
      gsl_vector_set(x,i+p.N_grid,-2);
    }
  cout<<2*p.N_grid<<endl;

  T = gsl_multiroot_fsolver_broyden;
  s = gsl_multiroot_fsolver_alloc (T, 2*p.N_grid);
  gsl_multiroot_fsolver_set (s, &f, x);
  //return 0;
  //return 0;
  print_state (iter, s);
  
  do
    {
      iter++;
      status = gsl_multiroot_fsolver_iterate (s);
      
      print_state (iter, s);
      
      if (status)   /* check if solver is stuck */
	break;
      
      status =
	gsl_multiroot_test_residual (s->f, 1e-7);
    }
  while (status == GSL_CONTINUE && iter < 1000);
  
  printf ("status = %s\n", gsl_strerror (status));
  
  gsl_multiroot_fsolver_free (s);
  gsl_vector_free (x);
  return 0;
}

