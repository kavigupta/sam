Generate evoluationary dataset:

python grab_sequence.py --splice_table evo_all.txt --ref_genome hg19.fa --sequence <sequence_file> --CL_max 400

python create_datafile.py --data_segment_to_use all --data_chunk_to_use all --splice_table evo_all.txt --data_dir <data_dir> --ref_genome hg19.fa --sequence <sequence_file> --CL_max 400

python create_dataset.py --data_segment_to_use all --data_chunk_to_use all --splice_table evo_all.txt --data_dir <data_dir> --ref_genome hg19.fa --sequence <sequence_file> --CL_max 400

Generate alternative dataset:

python grab_sequence.py --splice_table alt_all.txt --ref_genome hg19.fa --sequence <sequence_file> --CL_max 400

python create_datafile.py --data_segment_to_use all --data_chunk_to_use all --splice_table alt_all.txt --data_dir <data_dir> --ref_genome hg19.fa --sequence <sequence_file> --CL_max 400

python create_dataset.py --data_segment_to_use all --data_chunk_to_use all --splice_table alt_all.txt --data_dir <data_dir> --ref_genome hg19.fa --sequence <sequence_file> --CL_max 400