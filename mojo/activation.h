// == mojo ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files(the "Software"),
//    to deal in the Software without restriction, including without
//    limitation the rights to use, copy, modify, merge, publish, distribute,
//    sublicense, and/or sell copies of the Software, and to permit persons to
//    whom the Software is furnished to do so, subject to the following
//    conditions :
//
//    The above copyright notice and this permission notice shall be included
//    in all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ============================================================================
//    activation.h:  neuron activation functions
// ==================================================================== mojo ==

#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <cmath>
#include <algorithm>
#include <string>

namespace mojo {

	namespace relu
	{
		inline void  f(float *in, const int size, const float *bias)
		{
			for(int i=0; i<size; i++)
			{
				if((in[i] + bias[i]) < 0) in[i]= 0;
				else in[i]=(in[i] + bias[i]);
			}
		}
		inline void  fc(float *in, const int size, const float bias)
		{
			for(int i=0; i<size; i++)
			{
				if((in[i] + bias) < 0) in[i]= 0;
				else in[i]=(in[i] + bias);
			}
		}	inline float  df(float *in, int i, const int size) {if(in[i] > 0) return 1.0f; else return 0.0f; }
		const char name[]="relu";
	};

	namespace softmax
	{
		inline void f(float *in, const int size, const float *bias)
		{
			float max = in[0];
			for (int j = 1; j<size; j++) if (in[j] > max) max = in[j];

			float denom = 0;
			for (int j = 0; j<size; j++) denom += exp(in[j] - max);

			for(int i=0; i<size; i++) in[i]= exp(in[i] - max) / denom;
		}
		inline void fc(float *in, const int size, const float bias)
		{
			float max = in[0];
			for (int j = 1; j<size; j++) if (in[j] > max) max = in[j];

			float denom = 0;
			for (int j = 0; j<size; j++) denom += exp(in[j] - max);

			for(int i=0; i<size; i++) in[i]= exp(in[i] - max) / denom;
		}
		inline float df(float *in, int i, const int size)
		{
			// don't really use... should use good cost func to make this go away
			return in[i] * (1.f - in[i]);
		}

		const char name[] = "softmax";
	};

	typedef struct
	{
	public:
		void (*f)(float *, const int, const float*);
		void (*fc)(float *, const int, const float);
		float (*df)(float *, int, const int);
		const char *name;
	} activation_function;

	activation_function* new_activation_function(std::string act)
	{
		activation_function *p = new activation_function;
		if(act.compare(relu::name)==0) { p->f = &relu::f; p->fc = &relu::fc;p->df = &relu::df; p->name=relu::name;return p;}
		if(act.compare(softmax::name) == 0) { p->f = &softmax::f; p->fc = &softmax::fc;p->df = &softmax::df; p->name = softmax::name; return p; }
		delete p;
		return NULL;
	}

	activation_function* new_activation_function(const char *type)
	{
		std::string act(type);
		return new_activation_function(act);
	}

} // namespace

#endif // _ACTIVATION_H_
