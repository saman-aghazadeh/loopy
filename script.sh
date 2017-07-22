num_compute_units=1
num_simd=1
reqd_work_items="__attribute__((reqd_work_group_size(256,1,1)))"
done=0

while true
do
	grep -vE "(attribute)" $1.cl > temp/$1.cl
	sed -i "1s/^/$reqd_work_items\n/" temp/$1.cl
	sed -i "1s/^/__attribute__((num_simd_work_items($num_simd)))\n/" temp/$1.cl
	sed -i "1s/^/__attribute__((num_compute_units($num_compute_units)))\n/" temp/$1.cl
	#{ echo -n '$reqd_work_items '; cat temp/$1.cl } > temp/$1.cl
	#{ echo -n '__attribute__((num_simd_work_items($num_simd))) '; cat temp/$1.cl } > temp/$1.cl
	#{ echo -n '__attribute__((num_compute_units($num_compute_units))) '; cat temp/$1.cl } > temp/$1.cl
	aoc -c -v --board p385a_mac_ax115 temp/$1.cl -o bin/$1.aoco
	logicUtilization=`cat bin/$1/$1.log | grep -e "Logic utilization" | awk -F ' ' '{print substr($5, 0, length($5)-1)}'`
	ALUTs=`cat bin/$1/$1.log | grep -e "ALUTs" | awk -F ' ' '{print substr($4, 0, length($4)-1)}'`
	dedicated=`cat bin/$1/$1.log | grep -e "Dedicated logic registers" | awk -F ' ' '{print substr($6, 0, length($6)-1)}'`
	memblocks=`cat bin/$1/$1.log | grep -e "Memory blocks" | awk -F ' ' '{print substr($5, 0, length($5)-1)}'`
	dspblocks=`cat bin/$1/$1.log | grep -e "DSP blocks" | awk -F ' ' '{print substr($5, 0, length($5)-1)}'`

	echo "Logic Utilizations is $logicUtilization"
	echo "ALUTs is $ALUTs"
	echo "Dedicated is $dedicated"
	echo "Mem Blocks is $memblocks"
	echo "DSP blocks is $dspblocks"
	
	if [ $done -eq 1 ]; then	
		cat temp/$1.cl > $1.cl
		break
	fi

	if [ $logicUtilization -lt 100 ] && [ $ALUTs -lt 100 ] && [ $dedicated -lt 100 ] && [ $memblocks -lt 100 ] && [ $dspblocks -lt 100 ]; then
		if [ $num_simd != 16 ]; then
			let num_simd=$num_simd*2	
		else
			let num_compute_units=$num_compute_units+1
		fi
	else
		if [ $num_compute_units -gt 1 ]; then
			let num_compute_units=$num_compute_units-1
		else
			let num_simd=$num_simd/2
		fi
		let done=1	
	fi

	echo "num_simd $num_simd"
	echo "num_compute_units $num_compute_units" 

done
