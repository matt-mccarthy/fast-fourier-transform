out_dir="results"
trials=1000
arr_size=(4096 8192 16384 32768 65536 131072 262144 524288 1048576)
num_thds=(2 4 8 16 32)

seq_file="./sequential-fft"
par_file="./parallel-fft"

rm -rf $out_dir
mkdir $out_dir

for size in "${arr_size[@]}"
do
	out=$($seq_file $size $trials)
	echo -e $size'\t'$out >> $out_dir/1.txt
done

for threads in "${num_thds[@]}"
do
	exprans=`expr $threads \\* $threads`
	for size in "${arr_size[@]}"
	do
		out=$($par_file $size $trials $threads $threads)
		echo -e $size'\t'$out >> $out_dir/$exprans.txt
	done
done
