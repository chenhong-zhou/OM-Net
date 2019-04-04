#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/core_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CoreSoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
  weight_by_label_freqs_ =
    this->layer_param_.loss_param().weight_by_label_freqs();
  
  if (weight_by_label_freqs_) {
    vector<int> count_shape(1, this->layer_param_.loss_param().class_weighting_size());

    LOG(INFO)<< this->layer_param_.loss_param().class_weighting_size();
 
   label_counts_.Reshape(count_shape);
    CHECK_EQ(this->layer_param_.loss_param().class_weighting_size(), bottom[0]->channels())
		<< "Number of class weight values does not match the number of classes.";
    float* label_count_data = label_counts_.mutable_cpu_data();
    for (int i = 0; i < this->layer_param_.loss_param().class_weighting_size(); i++) {
        label_count_data[i] = this->layer_param_.loss_param().class_weighting(i);
    }
  }
  
   
}

template <typename Dtype>
void CoreSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_); //N
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);//H*W
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  if (weight_by_label_freqs_) {
    CHECK_EQ(this->layer_param_.loss_param().class_weighting_size(), bottom[0]->channels())
		<< "Number of class weight values does not match the number of classes.";
  }
  

  
}

template <typename Dtype>
void CoreSoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_; 
  int count = 0;
  Dtype loss = 0;
 
 
   const int batch_dim = bottom[0]->shape(0);
   vector<int> batch_dim_shape(1,batch_dim);
   batch_flag_.Reshape(batch_dim_shape);   
   float* batch_flag_data = batch_flag_.mutable_cpu_data();
   caffe_scal(batch_flag_.count(), float(0), batch_flag_data);
   
    for (int i = 0; i < outer_num_/2; ++i) {   
		int mm_count = 0;
        for (int j = 0; j < inner_num_; j++) {  
			const int label_value = label[i * inner_num_ + j];	
			if(label_value == 1||label_value == 2 || label_value == 3 )
			{  
				++mm_count;
			}  
		}
		if(mm_count >= 6553){  //(32*32*16)*0.5 = 8192   0.4:6553
			batch_flag_data[i] = 1.0;
		}
	}
	
	for (int i = outer_num_/2; i < outer_num_; ++i) {    
       batch_flag_data[i]=1.0; 
	}  

	for (int i = 0; i < outer_num_; ++i) {    

		if(batch_flag_data[i] == 1.0){
			for (int j = 0; j < inner_num_; j++) {
				const int label_value = static_cast<int>(label[i * inner_num_ + j]);
			if (has_ignore_label_ && label_value == ignore_label_) {
				continue;
			}
			DCHECK_GE(label_value, 0);
			DCHECK_LT(label_value, prob_.shape(softmax_axis_));
			const int idx = i * dim + label_value * inner_num_ + j;
			if (weight_by_label_freqs_) {
				const float* label_count_data = label_counts_.cpu_data();
				loss -= log(std::max(prob_data[idx], Dtype(FLT_MIN)))
				* static_cast<Dtype>(label_count_data[label_value]);
			} else {
				loss -= log(std::max(prob_data[idx], Dtype(FLT_MIN)));
			}
			++count;
			}
		}
	}
	
	if (normalize_) {
		top[0]->mutable_cpu_data()[0] = loss / count;
	} else {
		top[0]->mutable_cpu_data()[0] = loss / outer_num_;
	}
	if (top.size() == 2) {
		top[1]->ShareData(prob_);
	}
}

template <typename Dtype>
void CoreSoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* label = bottom[1]->cpu_data();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    int dim = prob_.count() / outer_num_;
    int count = 0;
    const float* label_count_data = 
        weight_by_label_freqs_ ? label_counts_.cpu_data() : NULL;
    
   const float* batch_flag_data = batch_flag_.cpu_data();   
			
    for (int i = 0; i < outer_num_; ++i) {
        if(batch_flag_data[i] == 1.0){		
		  for (int j = 0; j < inner_num_; ++j) {
			const int label_value = static_cast<int>(label[i * inner_num_ + j]);
			if (has_ignore_label_ && label_value == ignore_label_) {
			  for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
				bottom_diff[i * dim + c * inner_num_ + j] = 0;
			  }
			} else {
			  const int idx = i * dim + label_value * inner_num_ + j;
			  bottom_diff[idx] -= 1;
			  if (weight_by_label_freqs_) {
				for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
				  bottom_diff[i * dim + c * inner_num_ + j] *= static_cast<Dtype>(label_count_data[label_value]);
				}
			  }
			  ++count;
			}
		  }
		}else{
			for (int j = 0; j < inner_num_; ++j) {
				for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
					bottom_diff[i * dim + c * inner_num_ + j] = 0;
				}
			} 
		}
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CoreSoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(CoreSoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(CoreSoftmaxWithLoss);

}  // namespace caffe
