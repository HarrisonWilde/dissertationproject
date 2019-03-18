rm ../RefinedMidi/*
counter=0
for file1 in ../Audio2.nosync/*
do
	name1=$(basename -- "$file1")
	name1="${name1%.*}"
	found="false"
	for file2 in ../tfMidi/*
	do
		name2=$(basename -- "$file2")
		name2="${name2%.*}"
		if [[ $name1 == $name2 ]]
		then
			echo "Found $name1, copying..."
			scp "$file2" ../RefinedMidi/
			scp "../Midi2/16384 $name2.mid" ../RefinedMidi/
			scp "../Midi2/32768 $name2.mid" ../RefinedMidi/
			found="true"
		fi
	done
	if [[ $found == "false" ]]
	then
		for file2 in ../Midi2/*
		do
			name2=$(basename -- "$file2")
			name2="${name2%.*}"
			if [[ "16384 $name1" == "$name2" ]]
			then
				echo "Found $name1, copying..."
				scp "../Midi2/$name2.mid" ../RefinedMidi/
				scp "../Midi2/32768 $name1.mid" ../RefinedMidi/
				found="true"
			fi
		done
		if [[ $found == "false" ]]
		then
			echo "$name1 not found or copied"
			counter=$((counter + 1))
		fi
	fi
done

if [[ $counter > 0 ]]
then
	echo "$counter files were not found during execution."
done