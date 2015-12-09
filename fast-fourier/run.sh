out_dir="results"
trials=1000
# Must be powers of two.
arr_size=(4096 8192 16384 32768 65536 131072 262144 524288 1048576)
# Must also be powers of two. I pass this twice since I use num_blks=num_thds.
num_thds=(2 4 8 16 32)
# The names of the executables
seq_file="./sequential-fft"
par_file="./parallel-fft"
# Clean the result directory
rm -rf $out_dir
mkdir $out_dir
# Run sequential
for size in "${arr_size[@]}"
do
	out=$($seq_file $size $trials)
	echo -e $size'\t'$out >> $out_dir/1.txt
done
# Run parallel
for threads in "${num_thds[@]}"
do
	# Total number of threads = num_blk * num_thd
	exprans=`expr $threads \\* $threads`
	for size in "${arr_size[@]}"
	do
		out=$($par_file $size $trials $threads $threads)
		echo -e $size'\t'$out >> $out_dir/$exprans.txt
	done
done
