The production of the openei InSPIRE wiki widget requires that we map (lat, lon) pairs to an arbitrary gid index provided by the NSRDB.

We worked with Michael Sherman of the NSRDB team to create an API using their database to do this. 
This directory contains a query result from the database table the new API will pull from and a test script to verify that it matches the NSRDB source 
he dataset was calculated from on the Kestrel HPC system.
