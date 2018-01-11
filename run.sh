main_operation=$1
main_function=$2
main_data=$3
main_dict_num=$4
main_dict_thre=$5
main_dev_num=$6

if [ "$main_function" = "DeleteOnly" ]; then
main_function=orgin
elif [ "$main_function" = "DeleteAndRetrieve" ]; then
main_function=label
fi

if [ "$main_data" = "yelp" ]; then
main_dict_num=7000
main_dict_thre=15
main_dev_num=4000
elif [ "$main_data" = "imagecaption" ]; then
main_dict_num=3000
main_dict_thre=5
main_dev_num=1000
elif [ "$main_data" = "amazon" ]; then
main_dict_num=10000
main_dict_thre=5.5
main_dev_num=2000
fi

#configure
data_tool_path=data/tool/
model_file=data/style_transform_${main_function}/
train_data_file=data/$main_data/train.data.${main_function}.shuffle
dict_data_file=data/$main_data/zhi.dict.${main_function}
data_file_prefix=data/$main_data/${main_category}.
main_category=sentiment
main_category_num=2

if [ "$main_operation" = "train" ]; then
echo train

#preprocess train data
cd $data_tool_path 
sh preprocess_train_data.sh $main_operation $main_function $main_data $main_category $main_category_num $main_dict_num $main_dict_thre
cd ../../
line_num=$(wc -l < $train_data_file)
vt=$main_dev_num
eval $(awk 'BEGIN{printf "train_num=%.6f",'$line_num'-'$vt'}')
test_num=$main_dev_num
vaild_num=0
eval $(awk 'BEGIN{printf "train_rate=%.6f",'$train_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "vaild_rate=%.6f",'$vaild_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "test_rate=%.6f",'$test_num'/'$line_num'}')

#train process
rm -rf $model_file
THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu3,floatX=float32' python main.py ../$model_file ../$train_data_file ../$dict_data_file aux_data/stopword.txt aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT train

elif [ "$main_operation" = "test" ]; then
echo test

#preprocess test data
line_num=$(wc -l < $train_data_file)
vt=$main_dev_num
eval $(awk 'BEGIN{printf "train_num=%.6f",'$line_num'-'$vt'}')
test_num=$main_dev_num
vaild_num=0
eval $(awk 'BEGIN{printf "train_rate=%.6f",'$train_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "vaild_rate=%.6f",'$vaild_num'/'$line_num'}')
eval $(awk 'BEGIN{printf "test_rate=%.6f",'$test_num'/'$line_num'}')
cd $data_tool_path 
sh preprocess_test_data.sh $main_operation $main_function $main_data $main_category $main_category_num $main_dict_num $main_dict_thre
cd ../../
for((i=0;i<$main_category_num;i++))
do
	THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu3,floatX=float32' python main.py ../$model_file ../$train_data_file ../$dict_data_file aux_data/stopword.txt aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_emb ${data_file_prefix}train.${i}.template.${main_function}
        THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu3,floatX=float32' python main.py ../$model_file ../$train_data_file ../$dict_data_file aux_data/stopword.txt aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_emb ${data_file_prefix}test.${i}.template.${main_function}
done
cd $data_tool_path
for((i=0;i<$main_category_num;i++))
do
	python find_nearst_neighbot_all.py $i $main_data ${main_function}
	python form_test_data.py ../../${data_file_prefix}test.${i}.template.${main_function}.emb.result
	if [ "$main_function" = "entire" ]; then
		python change_content.py ../../${data_file_prefix}test.${i}.template.orgin ../../${data_file_prefix}test.${i}.template.${main_function}.emb.result.filter
		cp ../../${data_file_prefix}test.${i}.template.${main_function}.emb.result.filter.change  ../../${data_file_prefix}test.${i}.template.${main_function}.emb.result.filter		
	fi
done
cd ../../

#test process
for((i=0;i<$main_category_num;i++))
do 
THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu3,floatX=float32' python main.py ../$model_file ../$train_data_file ../$dict_data_file aux_data/stopword.txt aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderDT generate_b_v_t ${data_file_prefix}test.${i}.template.${main_function}.emb.result.filter
done
cd $data_tool_path
for((i=0;i<$main_category_num;i++))
do
	python build_lm_data.py ../../${data_file_prefix}train.${i}
	python shuffle.py ../../${data_file_prefix}train.${i}.lm
	python create_dict.py ../../${data_file_prefix}train.${i}.lm ../../${data_file_prefix}train.${i}.lm.dict
done
cd ../../
for((i=0;i<$main_category_num;i++))
do
	lm_model_file=data/style_transform_${main_function}_lm_${i}/
	rm -rf $lm_model_file
THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu2,floatX=float32' python main.py ../$lm_model_file ../${data_file_prefix}train.${i}.lm.shuffle ../${data_file_prefix}train.${i}.lm.dict aux_data/stopword.txt aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm train
done
i=0
lm_model_file=data/style_transform_${main_function}_lm_${i}/ 
THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu3,floatX=float32' python main.py ../$lm_model_file ../${data_file_prefix}train.${i}.lm.shuffle ../${data_file_prefix}train.${i}.lm.dict aux_data/stopword.txt aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm generate_b_v_t_v ${data_file_prefix}test.1.template.${main_function}.emb.result.filter.result
i=1
lm_model_file=data/style_transform_${main_function}_lm_${i}/
THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu3,floatX=float32' python main.py ../$lm_model_file ../${data_file_prefix}train.${i}.lm.shuffle ../${data_file_prefix}train.${i}.lm.dict aux_data/stopword.txt aux_data/embedding.txt $train_rate $vaild_rate $test_rate ChoEncoderDecoderLm generate_b_v_t_v ${data_file_prefix}test.0.template.${main_function}.emb.result.filter.result
cd $data_tool_path
for((i=0;i<$main_category_num;i++))
do
        python get_final_result.py ../../${data_file_prefix}test.${i}.template.${main_function}.emb.result.filter.result.result ${i}
	#cp ../../${data_file_prefix}test.${i}.template.${main_function}.emb.result.filter.result.result.final_result ../result_combine/${main_category}.test.${i}.${main_function}
done
cd ../../
fi
