

# Data sourcing

Downloaded from gcp bucket

```
https://console.cloud.google.com/storage/browser/_details/gtex-resources/GTEx_Analysis_v8_QTLs/GTEx_Analysis_v8_sQTL_leafcutter_counts.tar;tab=live_object?project=mineral-subject-307700
```

Used command (on kavi's desktop) 

```
gsutil -u mineral-subject-307700 cp gs://gtex-resources/GTEx_Analysis_v8_QTLs/GTEx_Analysis_v8_sQTL_leafcutter_counts.tar .
```
Then copy the file over to the server and run in this directory

```
tar xvf ~/GTEx_Analysis_v8_sQTL_leafcutter_counts.tar
```

