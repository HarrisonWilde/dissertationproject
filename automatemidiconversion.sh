for file in ../Audio.nosync/*
do
	filefull="$file"
	name=$(basename -- "$file")
	nameonly="${name%.*}"
	extension="${filefull##*.}"
	filename="${filefull%.*}"

	if [ ! -f "../Midi2/32768 $nameonly.mid" ]
	then
		if [ "$extension" != "wav" ]
		then
			sox "$filefull" "$filename.wav"
			rm "$filefull"
		fi
		../waon/waon -i "$filename.wav" -o "../Midi2/32768 $nameonly.mid" -n 32768
	else
		echo "32768 Midi already generated for $nameonly"
	fi
	
	if [ ! -f "../Midi2/16384 $nameonly.mid" ]
	then
		if [ "$extension" != "wav" ]
		then
			sox "$filefull" "$filename.wav"
			rm "$filefull"
		fi
		../waon/waon -i "$filename.wav" -o "../Midi2/16384 $nameonly.mid" -n 16384
	else
		echo "16384 Midi already generated for $nameonly"
	fi
	
	if [ ! -f "../Midi2/8192 $nameonly.mid" ]
	then
		if [ "$extension" != "wav" ]
		then
			sox "$filefull" "$filename.wav"
			rm "$filefull"
		fi
		../waon/waon -i "$filename.wav" -o "../Midi2/8192 $nameonly.mid" -n 8192
	else
		echo "8192 Midi already generated for $nameonly"
	fi
done