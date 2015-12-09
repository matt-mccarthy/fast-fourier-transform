out_dir="results"
trials=1000
arr_size=(4096 8192 16384 32768 65536 131072)
num_thds=("4 4" "8 8" "16 16" "32 32")

seq_file="sequential-fft"
par_file="parallel-fft"

rm -rf $out_dir
mkdir $out_dir

for size in "${arr_size[@]}"
do
	out=$($seq_file $threads $size $trials)
	echo -e $size'\t'$out >> $out_dir/1.txt
done

for threads in "${num_thds[@]}"
do
	for size in "${arr_size[@]}"
	do
		out=$($par_file $threads $size $trials)
		echo -e $size'\t'$out >> $out_dir/$threads.txt
	done
done
