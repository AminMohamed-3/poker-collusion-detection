import sys; sys.dont_write_bytecode=True
import pyarrow.parquet as pq
base="C:/Data/Projects/Money Making/work/poker/data/"
for f in ["hands","actions","seats","players"]:
    pf=pq.ParquetFile(base+f+".parquet")
    print("=====",f,"rows",pf.metadata.num_rows,"rowgroups",pf.metadata.num_row_groups)
    print(pf.schema_arrow)
