# Run like so:
#    LC_NUMERIC="en_UK.UTF-8" awk -f <input file>
# in case locale is not standard US/UK

BEGIN {
    FS = ";"
    if (ENVIRON["LC_NUMERIC"] == "nb_NO.UTF-8") {
        print "Incompatible locale; run with"
	print "  LC_LOCALE=en_US.UTF-8; awk -f summarize_results.awk <input file>"
	exit 1
    }
}
{
    # Header
    if (NR == 1) {
        for (i=1; i<=NF; i++) {
            colnum_to_header[i] = $i
            results[i] = 0.0
	    #print "colnum_to_header[" i "]:", $i
        }
    }
    # Data
    else {
        for (i=1; i<=NF; i++) {
            results[i] += $i
        }
    }
}
END {
    num_entries = NR-1 
    for (k=1; k<=length(results); k++) {
        avg = results[k]/num_entries
        print colnum_to_header[k], avg 
    }
}
