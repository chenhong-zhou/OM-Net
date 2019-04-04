#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/enh_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EnhAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.enh_accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.enh_accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.enh_accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void EnhAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
      << "top_k must be less than or equal to the number of classes.";
  label_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.enh_accuracy_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(label_axis_);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void EnhAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int dim = bottom[0]->count() / outer_num_;
  const int num_labels = bottom[0]->shape(label_axis_);
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  
  
  label_fuben_.ReshapeLike(*bottom[1]);
  Dtype* label_fuben   = label_fuben_.mutable_cpu_data();
  caffe_copy(outer_num_ * inner_num_, bottom_label, label_fuben);  
  
    const int batch_dim = bottom[0]->shape(0);
   vector<int> batch_dim_shape(1,batch_dim);
   batch_flag_.Reshape(batch_dim_shape);   
   float* batch_flag_data = batch_flag_.mutable_cpu_data();
   caffe_scal(batch_flag_.count(), float(0), batch_flag_data);
  
  for (int i = 0; i < outer_num_/3 *2; ++i) {   //batch_flag_data
    int mm_count = 0;
    for (int j = 0; j < inner_num_; j++) {  
      const int label_value = bottom_label[i * inner_num_ + j];	 
	  if(label_value == 3 ||label_value == 1 )
	  {    
           ++mm_count;
	  }  
	  if(label_value == 4 ||label_value == 0 ||label_value == 2  )
	  {
		  label_fuben[i * inner_num_ + j] = 0;
	  }
	  if(label_value == 3 )
	  {
		  label_fuben[i * inner_num_ + j] = 1;
	  }	  
	  if(label_value == 1 )
	  {
		  label_fuben[i * inner_num_ + j] = 0;
	  }	 
	}
	if(mm_count >= 8192){  //(32*32*16)*0.5 = 8192  
		batch_flag_data[i] = 1.0;
	}
  }
  for (int i = outer_num_/3 *2; i < outer_num_; ++i) {    
       batch_flag_data[i]=1.0; 
  
  }   
  
  
  int count = 0;
 for (int i = 0; i < outer_num_; ++i) {
   if(batch_flag_data[i] == 1.0){	  
    for (int j = 0; j < inner_num_; ++j) {
		
	   int label_value_yuan =static_cast<int>(bottom_label[i * inner_num_ + j]);  
	   int label_value = label_value_yuan; 
	  if(i< outer_num_/3 *2) 
	  { 
        if(label_value == 4 ||label_value == 0 ||label_value == 2 || label_value == 1  )
	    {
		   label_value = 0;
	    }
	    if(label_value == 3 )
	    {
		   label_value = 1;
	    }	
	  }

      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, num_labels);
      // Top-k accuracy
      std::vector<std::pair<Dtype, int> > bottom_data_vector;
      for (int k = 0; k < num_labels; ++k) {
        bottom_data_vector.push_back(std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k));
      }
      std::partial_sort(
          bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
      // check if true label is in top k predictions
      for (int k = 0; k < top_k_; k++) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
          break;
        }
      }
      ++count;
    }
	
  }
  }

  
  top[0]->mutable_cpu_data()[0] = accuracy / count;
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
          nums_buffer_.cpu_data()[i] == 0 ? 0
          : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
    }
  }
  // EnhAccuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(EnhAccuracyLayer);
REGISTER_LAYER_CLASS(EnhAccuracy);

}  // namespace caffe
