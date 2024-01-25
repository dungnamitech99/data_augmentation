if [ $# -ne 3 ]; then
    echo $0 " <db1> <db2> <outdb"
    exit 1
fi

db1=$1
db2=$2
outdb=$3

mkdir -p $outdb

for f in calib.txt  labels  label_time_wav.txt  uttid  wav.scp; do
    cat $db1/$f $db2/$f > $outdb/$f
done
