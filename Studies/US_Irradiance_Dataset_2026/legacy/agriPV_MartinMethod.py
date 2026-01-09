import numpy as np
import os
import pandas as pd
import time
import math
from itertools import chain
from itertools import product
import bifacial_radiance as br
from dask.distributed import Client, LocalCluster, secede
import math
import datetime
from timeit import default_timer as timer
from time import sleep
from rex import NSRDBX
import pytz
import pickle
import bifacialvf
import shutil
# import subprocess

# rc = subprocess.call("/home/etonita/BasicSimulations/start_script.sh")

def start_dask(hpc=None):
    """
    Starts a dask cluster for parallel processing.

    Parameters
    ----------
    hpc : dict
        Dictionary containing dask hpc settings (see examples below).

    Examples
    --------
    Local cluster:

    .. code-block:: python

        hpc = {'manager': 'local',
               'n_workers': 1,
               'threads_per_worker': 8,
               'memory_limit': '10GB'}

    SLURM cluster:

    .. code-block:: python

        kestrel = {
            'manager': 'slurm',
            'n_jobs': 1,  # Max number of nodes used for parallel processing
            'cores': 104,
            'memory': '256GB',
            'account': 'pvsoiling',
            'walltime': '4:00:00',
            'processes': 52,
            'local_directory': '/tmp/scratch',
            'job_extra_directives': ['-o ./logs/slurm-%j.out'],
            'death_timeout': 600,}

    Returns
    -------
    client : dask.distributed.Client
        Dask client object.
    """
    if hpc is None:
        cluster = LocalCluster()
    else:
        manager = hpc.pop('manager')

        if manager == 'local':
            cluster = LocalCluster(**hpc)
        elif manager == 'slurm':
            from dask_jobqueue import SLURMCluster
            n_jobs = hpc.pop('n_jobs')
            cluster = SLURMCluster(**hpc)
            cluster.scale(jobs=n_jobs)

    client = Client(cluster)
    print('Dashboard:', client.dashboard_link)
    client.wait_for_workers(n_workers=1)

    return client


# Run simulation using the given timestamp and wavelength
def simulate_single(df_tmy = None, meta_dict = None, gid = None, setup = None,
             startdate=None, rootPath=None):

    startdatestr = str(startdate).replace(':','_').replace(' ','__')

    #startdate = None
    #enddate = None
    simpath = f'{state}_{gid}_setup_{setup}_{startdatestr}'
    if rootPath is None:
        path = os.path.join(state,str(setup),simpath)
    else:
        path = os.path.join(rootPath,state,str(setup),simpath)
    results_path = os.path.join(path, 'results.pkl')

    #os.path.isfile(path)
    #Check if simulation has already completed
    print("Current path", path)

    if os.path.isfile(results_path):
        print("***** SIM Done for ", simpath)
        print("Results are: ", results)
        
        return 1

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    alb = 0.2
    radObj = br.RadianceObj(simpath,path, hpc=True)
    radObj.setGround(alb) 

    enddate = startdate + datetime.timedelta(hours=23)
    print("Meta_dict into readWeatherData")
    metData = radObj.readWeatherData(metadata=meta_dict, metdata=df_tmy, starttime=startdate,
                                      endtime=enddate, coerce_year=2023,
                                      label='center')
    
    # Tracker Projection of half the module into the ground, 
    # for 1-up module in portrait orientation
    # Assuming still 1 m for fixed-tilt systems even if a bit less 
    # VERTICAL setup changes xp to just a margin 20 cm for protection of the modules.
    xp = 10 
    y = 2
    solposAzi = metData.solpos['azimuth'][0]
    solposZen = metData.solpos['zenith'][0]
    timezonesave = metData.timezone

    tilt = np.round(metData.latitude)
    if tilt > 40.0:
        tilt = 40.0
    
    DD = bifacialvf.vf.rowSpacing(beta = tilt, 
                                  sazm=180, lat = metData.latitude, 
                                  lng = metData.longitude, 
                                  tz = metData.timezone, 
                                  hour = 9, 
                                  minute = 0.0)
    if (DD <= 0) | (DD > 3.725):
        DD = 3.725
        print("Cannot find ideal pitch for location, setting D to 3.725")
    # Set minimum of 2.0 m inter-row spacing (accounting for y = 2)
    if (DD <= 1.0):
        DD = 1.0

    normalized_pitch = DD + np.cos(np.round(metData.latitude) / 180.0 * np.pi)
    pitch_temp = normalized_pitch*y
    print("PITCH_temp ", pitch_temp)

    if setup == 1:
        hub_height = 1.5
        pitch = 5
        sazm = 180  # Tracker axis azimuth
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = None
    if setup == 2:
        hub_height = 2.4
        pitch = 5
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = None
    if setup == 3:
        hub_height = 2.4
        pitch = 5
        sazm = 180
        modulename = 'PVmodule_1mxgap'
        bedsWanted = 3
        fixed_tilt_angle = None
    if setup == 4:
        hub_height = 1.5
        pitch = 8
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 6
        fixed_tilt_angle = None
    if setup == 5:
        hub_height = 1.5
        pitch = 11
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 9
        fixed_tilt_angle = None
    if setup == 6:
        hub_height = 1.5
        #tilt = None # fixed
        sazm = 180
        pitchfactor = 1
        modulename = 'PVmodule'
        pitch = pitch_temp * pitchfactor
        bedsWanted = 3
        fixed_tilt_angle = tilt
    if setup == 7:
        hub_height = 2.4 
        sazm = 180
        pitchfactor = 1
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = tilt
    if setup == 8:
        hub_height = 2.4 
        sazm = 180
        pitchfactor = 1
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule_1mxgap'
        bedsWanted = 3
        fixed_tilt_angle = tilt
    if setup == 9:
        hub_height = 1.5 
        sazm = 180
        pitchfactor = 2
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule'
        bedsWanted = 6
        fixed_tilt_angle = tilt
    if setup == 10:
        hub_height = 1.6 
        sazm = 90
        pitch = 8.6 
        modulename = 'PVmodule'
        bedsWanted = 7
        xp = 8
        fixed_tilt_angle = 90

    # TILT & PITCH CALCULATION HERE

    gcr = 2/pitch

    # -- establish tracking angles
    trackerParams = {'limit_angle':50,
                     'angledelta':5,
                     'backtrack':True,
                     'gcr':gcr,
                     'cumulativesky':False,
                     'azimuth': sazm,
                     'fixed_tilt_angle': fixed_tilt_angle,
                     }

    trackerdict = radObj.set1axis(**trackerParams)
    
    # -- generate sky   
    trackerdict = radObj.gendaylit1axis()

    sceneDict = {'pitch':pitch, 
                 'hub_height': hub_height,
                 'nMods': 19,
                 'nRows': 7,
                'tilt': fixed_tilt_angle,  
                'sazm': sazm
                 }

    modWanted = 10
    rowWanted = 4

    trackerdict = radObj.makeScene1axis(module=modulename,sceneDict=sceneDict)

    # -- build oct file
    trackerdict = radObj.makeOct1axis()

    # -- run analysis
    # Analysis for Module
    trackerdict = radObj.analysis1axis(customname = 'Module',
                                       sensorsy=9, modWanted=modWanted,
                                       rowWanted=rowWanted)

    try:
        trackerdict = radObj.calculateResults(bifacialityfactor=0.7, agriPV=False)
    except:
        print("**Error on trackerdict WHY!, skipping", gid, startdate)
        print("Trackerdict error path: " , results_path)
        print("TRACKERDICT Print:", radObj.trackerdict)      
        return 0

    ResultDatetime = list(radObj.CompiledResults['timestamp']) 
    ResultPVWm2Back = list(radObj.CompiledResults['Grear_mean'])
    ResultPVWm2Front = list(radObj.CompiledResults['Gfront_mean'])
    ResultGHI = list(radObj.CompiledResults['GHI'])
    ResultDNI = list(radObj.CompiledResults['DNI'])
    ResultDHI = list(radObj.CompiledResults['DHI'])
    ResultPout = list(radObj.CompiledResults['Pout'])
    # Commenting out due to using version 0.4.2+236.g6801d3d
    # ResultModuleTemp = list(radObj.CompiledResults['Module_temp'])
    ResultWindSpeed = list(radObj.CompiledResults['Wind Speed'])

    # Modify modscanfront for Ground
    resolutionGround = 0.1  # use 1 for faster test runs
    numsensors = int((pitch/resolutionGround)+1)
    modscanfront = {'xstart': 0, 
                    'zstart': 0.05,
                    'xinc': resolutionGround,
                    'zinc': 0,
                    'Ny':numsensors,
                    'orient':'0 0 -1'}

    # Analysis for GROUND
    trackerdict = radObj.analysis1axis(customname = 'Ground',
                                       modWanted=modWanted, rowWanted=rowWanted,
                                        modscanfront=modscanfront, sensorsy=1)
 
    keys=list(trackerdict.keys())

    ResultGroundIrrad = []
    ResultTemp = []
    for key in keys:
        ResultGroundIrrad.append(list(trackerdict[key]['AnalysisObj'][1].Wm2Front))
        ResultTemp.append(trackerdict[key]['temp_air'])

    # Cleanup of Front files from the Ground simulation
    filesall = os.listdir('results')
    filestoclean = [e for e in filesall if e.endswith('_Back.csv')]
    for cc in range(0, len(filestoclean)):
        filetoclean = filestoclean[cc]
        os.remove(os.path.join('results', filetoclean))

    results = pd.DataFrame(list(zip(ResultDatetime, ResultGHI, ResultDNI, ResultDHI, 
                                    ResultTemp, ResultWindSpeed, 
                                    ResultPVWm2Back, ResultPVWm2Front,
                                    ResultPout, ResultGroundIrrad #ResultModuleTemp, 
                                    )), 
                                    columns=["Timestamp", "GHI", "DNI", "DHI", "AirTemp", 
                                             "WindSpeed", "PVWm2Back", "PVWM2Front",
                                                  "PVPowerOutputW", "GroundIrrad" # "PVModuleTemp",
                                                  ])
    results["gid"] = gid
    results["setup"] = setup
    results["latitude"] = metData.latitude
    results["longitude"] = metData.longitude

    results["pitch"] = trackerdict[key]['scenes'][0].sceneDict['pitch']

    # save to folder    
    results.to_pickle(results_path)
    print("Results pickled!")

    # if os.path.isfile(results_path):
    #     # Verifies CSV file was created, then deletes unneeded files.
    #     for clean_up in os.listdir(path):
    #         if not clean_up.endswith('results.pkl'):
    #             clean_upfile = os.path.join(path, clean_up)
    #             if os.path.isfile(clean_upfile):    
    #                 os.remove(clean_upfile)
    #             else:
    #                 shutil.rmtree(clean_upfile)
    print("Results len ", len(results), " type ", type(results))
    print("All other files cleaned!")

    print("***** SIM Done for ", simpath, len(results), " \n Results: ", results)

    return 1


def run_simulations_dask(df_weather, meta, startdates, 
                         setups, rootPath, state, hpc):
    # Create client

    client = start_dask(hpc)
    
    print("dask client started")

    # Iterate over inputs
    futures = []
    

    # Add Iterations HERE
        #loop through dataframe and perform computation
    for setup in setups:
        for dd in range(0, len(startdates)):
            #print("setup: ", setup, ", startdate: ", dd)
            startdate = startdates[dd]
            for gid, row in meta.iterrows():
                #prepare input for PVDegTools
                meta_dict = row.to_dict()
                df_tmy = df_weather.loc[:, gid]
                tz_convert_val = meta_dict['timezone']
                df_tmy = df_tmy.tz_convert(pytz.FixedOffset(tz_convert_val*60))
                df_tmy.index =  df_tmy.index.map(lambda t: t.replace(year=2021)) 
                df_tmy = df_tmy.sort_index()

                debug = False
                if debug:
                    df_tmy.to_pickle('df_convert_'+str(gid)+'.pkl')
                    filesavepic = 'meta_convert_'+str(gid)+'.pkl'
                    with open(filesavepic, "wb") as ffp:   # pickling
                        pickle.dump(meta_dict,ffp)
                    print("Meta dict", meta_dict)
                futures.append(client.submit(simulate_single, df_tmy=df_tmy, 
                                             meta_dict=meta_dict, gid=gid,
                                             setup=setup, 
                                             startdate=startdate, 
                                             rootPath=rootPath)) 



    # Get results for all simulations
    res = client.gather(futures)

    # Close client
    client.close()

    res = 'FINISHED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    return res


if __name__ == "__main__":

    print(">>>>>>>>>>>>>>>>>> STARTING HERE !")
    print(datetime.datetime.now())
    sim_start_time=datetime.datetime.now()
    
    FullYear = False
    
    if FullYear:
        # start = datetime.strptime("01-01-2023", "%d-%m-%Y")
        start = datetime.datetime(2023, 1, 1, 0, 0)
        # end = datetime.strptime("31-12-2023", "%d-%m-%Y")
        end = datetime.datetime(2023, 12, 31, 0, 0)
    else:
        # start = datetime.strptime("01-11-2023", "%d-%m-%Y")
        # end = datetime.strptime("02-11-2023", "%d-%m-%Y")
        start = datetime.datetime(2023, 12, 13, 0, 0)
        end = datetime.datetime(2023, 12, 14, 0, 0)
    # date_generated = [start + timedelta(days=x) for x in range(0, (end-start).days)]
    # daylist = []
    # for date in date_generated:
    #     daylist.append(date.strftime("%y_%m_%d"))
    # loop doesn't add last day :
    # daylist.append('23_12_31')
    daylist = []
    while start <= end:
        daylist.append(start)
        start += datetime.timedelta(days=1)

    local = {'manager': 'local',
        'n_workers': 32,
        'threads_per_worker': 1, # Number of CPUs
        }
    kestrel = {
        'manager': 'slurm',
        'n_jobs': 1, # 4,  # Number of nodes used for parallel processing #1
        'cores': 100, #This is the total number of threads in all workers
        'memory': '256GB',
        'account': 'inspire',
        'queue': 'debug', #'standard'
        'walltime': '1:00:00',  #'8:00:00'
        'processes': 25, #This is the number of workers (each thread has more CPUs with a 100 cores:25 thread ratio)
        # Can experiment with this ratio to see what works
        #'interface': 'lo'
        #'job_extra_directives': ['-o ./logs/slurm-%j.out'],
        # This is useful if you get some sort of "could not connect to the Dask nanny" error
        #'death_timeout': 600
        # Use this if having memory issues
        # 'local_directory': '$TMPDIR',
        }

    now=datetime.datetime.now()
    results_path = "/scratch/kdoubled/test"+"_"+now.strftime('%Y-%m-%d_%Hh%M')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    smallSim = True # one location

    print("Bifacial_radiance version ", br.__version__)
    print("Pandas version ", pd.__version__)
    print("Numpy version ", np.__version__)

    starttimer = timer()

    #rootPath = r'/scratch/sayala/AgriDebugUS_Aug4'
    # rootPath = os.getcwd()
    nsrdb_file = '/kfs2/datasets/NSRDB/current/nsrdb_tmy-2021.h5'
    #TMY data located on eagle about 900GB

    #Input
    parameters = ['dhi', 'ghi', 'dni', 'air_temperature',  'wind_speed', 'wind_direction', 'surface_albedo', 
                  'dew_point', 'surface_pressure']

    with NSRDBX(nsrdb_file, hsds=False) as f:
        meta = f.meta
   
    print("NSRDB accessed")

    # meta_USA = meta[meta['country'] == 'Puerto Rico']
    meta_USA = meta[meta['country'] == 'United States']

    # startdates = []
    # with open('/home/sayala/AgriDebug/startdates_CO.txt', 'r') as fp:
    #     for line in fp:
    #         # remove linebreak from a current name
    #         # linebreak is the last character of each line
    #         x = line[:-1]

    #         # add current item to the list
    #         startdates.append(x)

    def gid_downsampling(meta, n):
        lon_sub = sorted(meta['longitude'].unique())[0:-1:max(1,2*n)]
        lat_sub = sorted(meta['latitude'].unique())[0:-1:max(1,2*n)]
        lon_sub2 = sorted(meta['longitude'].unique())[1:-1:max(1,2*n)]
        lat_sub2 = sorted(meta['latitude'].unique())[1:-1:max(1,2*n)]
        gids_sub = meta[(meta['longitude'].isin(lon_sub)) & (meta['latitude'].isin(lat_sub))].index
        meta_sub = meta.loc[gids_sub]
        return meta_sub, gids_sub

    if smallSim:
        meta_USA = meta_USA.loc[[219559,219563]]
        nsampling = 3
        setups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
        # startdates = [pd.to_datetime(i) for i in startdates]
        #startdates = startdates[3:6]
    else:
        meta_USA = meta_USA[meta_USA['state'] == 'Oregon']
        nsampling = 2 # 2
        setups = [1, 2] #, 3, 4, 5] #, 6, 7, 8, 9, 10] 
        startdates = [pd.to_datetime(i) for i in startdates]

    # Create client
    # client = Client(scheduler_file=scheduler_file)
    
    #region_col = 'country' 
    region_col = 'state' # 'state'



    for state in meta_USA[region_col].unique():
        region = state
        #Load time and geographical infos
        with NSRDBX(nsrdb_file, hsds=False) as f:
            # Get time index
            times = f.time_index
            # Get geographical index for region of interest
            gids = f.region_gids(region=region, region_col=region_col)   
            # Get meta data
            meta = f.meta[f.meta.index.isin(gids)]

        if smallSim:
            gids_sub = [219559,219563] 
            meta2 = meta_USA
        else:
            meta2, gids_sub = gid_downsampling(meta, n=nsampling)


        data = []
        with NSRDBX(nsrdb_file, hsds=False) as f:
            for p in parameters:
                data.append(f.get_gid_df(p, np.array(list(gids_sub)))) #.values 

        print("GIDs appended")

        #Create multi-level dataframe
        columns = pd.MultiIndex.from_product([parameters, np.array(list(gids_sub))], names=["par", "gid"])
        df_weather = pd.concat(data, axis=1)
        df_weather.columns = columns
        df_weather = df_weather.swaplevel(axis=1).sort_index(axis=1)

        #meta = meta.iloc[::500] # Downsampling every 10 entries...
        #meta2 = meta2.iloc[1:3] # Downsampling every 10 entries...

        # Saving list of GIDs sent to simulate for sanity checking
        # with open('RUN_'+str(n)+'_'+state+'.pkl', "wb") as ffp:   # Unpickling
        #    pickle.dump(list(gids_sub),ffp)
        # df_weather.to_pickle('df_weather.pkl')

        # Pass variables being looped on, and kwargs
        run_simulations_dask(df_weather = df_weather, meta = meta2, 
                             setups = setups, startdates = daylist, 
                             rootPath = results_path, state = state, hpc=kestrel)


        print("*********** DONE ************", state)
        stop = timer()
        runTime = round(stop-starttimer,2)

        #with open('TIMER_'+str(n)+'_'+state+'.pkl', "wb") as tfp:   # Unpickling
        #    pickle.dump(runTime,tfp)

    # client.shutdown()

    # =========== Perform Simulation Set ===========
    stop = timer()
    runTime = round(stop-starttimer,2)
    min = int(runTime//60)
    sec = int(round((runTime/60 - min)*60,0))
    print('=======================================')
    print(f'Simulation Run Time: {min:02}:{sec:02}')
    print('=======================================')
    
    with open('TIMER_'+str(nsampling)+'_'+'ALL.pkl', "wb") as tfp:   # Unpickling
        pickle.dump(runTime,tfp)
    #compile(rootPath)

    # Puerto rico (~472 locations), 1 day (13 timestamps) for 5 setups, with about 1/3 of it already modeled
    # took 70 minutes in 10 nodes
    #


