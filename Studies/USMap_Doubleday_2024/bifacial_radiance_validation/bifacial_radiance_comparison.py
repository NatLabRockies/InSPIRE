import numpy as np
import os
import pandas as pd
import time
import math
from itertools import chain
from itertools import product
import bifacial_radiance as br
import math
import datetime
from timeit import default_timer as timer
from time import sleep
from rex import NSRDBX
import pytz
import pickle
import bifacialvf
import shutil
import pvdeg
import argparse

def optimal_gcr_pitch(latitude: float, cw: float = 2) -> tuple[float, float]:
    """
    determine optimal gcr and pitch for fixed tilt systems according to latitude and optimal GCR parameters for fixed tilt bifacial systems.

    .. math::

        GCR = \frac{P}{1 + e^{-k(\alpha - \alpha_0)}} + GCR_0

    Inter-row energy yield loss 5% Bifacial Parameters:

    +-----------+--------+-----------+
    | Parameter | Value  | Units     |
    +===========+========+===========+
    | P         | 0.560  | unitless  |
    | K         | 0.133  | 1/°       |
    | α₀        | 40.2   | °         |
    | GCR₀      | 0.70   | unitless  |
    +-----------+--------+-----------+

    Parameters
    ------------
    latitude: float
        latitude [deg]
    cw: float
        collector width [m]

    Returns
    --------
    gcr: float
        optimal ground coverage ratio [unitless]
    pitch: float
        optimal pitch [m]

    References
    -----------
    Erin M. Tonita, Annie C.J. Russell, Christopher E. Valdivia, Karin Hinzer,
    Optimal ground coverage ratios for tracked, fixed-tilt, and vertical photovoltaic systems for latitudes up to 75°N,
    Solar Energy,
    Volume 258,
    2023,
    Pages 8-15,
    ISSN 0038-092X,
    https://doi.org/10.1016/j.solener.2023.04.038.
    (https://www.sciencedirect.com/science/article/pii/S0038092X23002682)

    Optimal GCR from Equation 4 
    Parameters from Table 1
    """

    p = -0.560 
    k = 0.133 
    alpha_0 = 40.2 
    gcr_0 = 0.70 

    # optimal gcr
    gcr = ((p) / (1 + np.exp(-k * (latitude - alpha_0)) )) + gcr_0

    pitch = cw / gcr
    return gcr, pitch

def inspire_practical_pitch(latitude: float, cw: float) -> tuple[float, float, float]:
    """
    Calculate pitch for fixed tilt systems for InSPIRE Agrivoltaics Irradiance Dataset.

    We cannot use the optimal pitch due to certain real world restrictions so we will apply some constraints.

    We are using latitude tilt but we cannot use tilts > 40 deg, due to racking constraints, cap at 40 deg for latitudes above 40 deg.

    pitch minimum: 3.8 m 
    pitch maximum:  12 m

    tilt max: 40 deg (latitude tilt)

    Parameters
    ----------
    latitude: float
        latitude [deg]
    cw: float
        collector width [m]

    Returns
    -------
    tilt: float
        tilt for a fixed tilt system with practical considerations [deg]
    pitch: float
        pitch for a fixed tilt system with practical consideration [m] 
    gcr: float
        gcr for a fixed tilt system with practical considerations [unitless]
    """

    gcr_optimal, pitch_optimal = optimal_gcr_pitch(latitude=latitude, cw=cw)

    pitch_ceil = min(pitch_optimal, 12)    # 12 m pitch ceiling
    pitch_practical = max(pitch_ceil, 3.8) # 3.8m pitch floor

    if not (3.8 <= pitch_practical <= 12):
        raise ValueError("calculated practical pitch is outside range [3.8m, 12m]")

    tilt_practical = min(latitude, 40)

    # practical gcr from practical pitch
    gcr_practical = cw / pitch_practical

    return float(tilt_practical), float(pitch_practical), float(gcr_practical)

# Run simulation using the given date, setup, and gid
def simulate_single(df_tmy = None, meta_dict = None, gid = None, setup = None,
             startdate=None, rootPath=None):

    startdatestr = str(startdate).replace(':','_').replace(' ','__')

    #startdate = None
    #enddate = None
    simpath = f'{gid}_setup_{setup}_{startdatestr}'
    if rootPath is None:
        path = os.path.join(str(setup),simpath)
    else:
        path = os.path.join(rootPath,str(setup),str(gid),simpath)
    results_path = os.path.join(path, 'results.pkl')

    #os.path.isfile(path)
    #Check if simulation has already completed
    print("Current path", path)

    if os.path.isfile(results_path):
        print("***** SIM Done for ", simpath)
        print("Results are: ", results)
        
        return 1

    # Create directory if it doesn't exist (exist_ok handles race conditions)
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
    solposAzi = metData.solpos['azimuth'].iloc[0]
    solposZen = metData.solpos['zenith'].iloc[0]
    timezonesave = metData.timezone

    # Practical tilt and pitch using Tonitas Equation
    # collector width of 2m for the inspire scenarios
    tilt, pitch_temp, gcr = inspire_practical_pitch(latitude=metData.latitude, cw=2)

    if setup == 1:
        hub_height = 1.5
        pitch = 5
        sazm = 180  # Tracker axis azimuth
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = None
        clearance_height = None
    if setup == 2:
        hub_height = 2.4
        pitch = 5
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = None
        clearance_height = None
    if setup == 3:
        hub_height = 2.4
        pitch = 5
        sazm = 180
        modulename = 'PVmodule_1mxgap'
        bedsWanted = 3
        fixed_tilt_angle = None
        clearance_height = None
    if setup == 4:
        hub_height = 1.5
        pitch = 8
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = None
        clearance_height = None
    if setup == 5:
        hub_height = 1.5
        pitch = 11
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 6
        fixed_tilt_angle = None
        clearance_height = None
    if setup == 6:
        hub_height = None
        sazm = 180
        pitchfactor = 1
        modulename = 'PVmodule'
        pitch = pitch_temp * pitchfactor
        bedsWanted = 3
        fixed_tilt_angle = tilt
        clearance_height = 1.5   
    if setup == 7:
        hub_height = None
        sazm = 180
        pitchfactor = 1
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = tilt
        clearance_height = 2.4
    if setup == 8:
        hub_height = None
        sazm = 180
        pitchfactor = 1
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule_1mxgap'
        bedsWanted = 3
        fixed_tilt_angle = tilt
        clearance_height = 2.4
    if setup == 9:
        hub_height = None
        sazm = 180
        pitchfactor = 2
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = tilt
        clearance_height = 1.5
    if setup == 10:
        hub_height = None
        sazm = 90
        pitch = 8.6 
        modulename = 'PVmodule'
        bedsWanted = 6
        xp = 8
        fixed_tilt_angle = 90
        clearance_height = 0.6

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

    if hub_height is not None:
        sceneDict = {'pitch':pitch, 
                    'hub_height': hub_height,
                    'clearance_height': clearance_height,
                    'nMods': 19,
                    'nRows': 7,
                    'tilt': fixed_tilt_angle,  
                    'sazm': sazm
                    }
    else: # Need to skip hub_height for fixed tilt
        sceneDict = {'pitch':pitch, 
                    'clearance_height': clearance_height,
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
    # Absolute coordinate system (west left)
    # West to East for single-axis tracker
    numsensors = 10
    resolutionGround = pitch / numsensors
    modscanfront = {'xstart': resolutionGround / 2, 
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
        try:
            ResultGroundIrrad.append(list(trackerdict[key]['AnalysisObj'][1].Wm2Front))
        except Exception as e:
            # Pad with NaN to match expected number of sensors, needed when sun altitude recalculation is nan
            ResultGroundIrrad.append([np.nan] * numsensors)
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


def run_simulations(df_weather, meta, startdates, 
                         setups, rootPath):

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
                # Date needs to be 2024 to match the NREL TMY-2024 data
                df_tmy.index =  df_tmy.index.map(lambda t: t.replace(year=2024))
                df_tmy = df_tmy.sort_index()

                debug = False
                if debug:
                    df_tmy.to_pickle('df_convert_'+str(gid)+'.pkl')
                    filesavepic = 'meta_convert_'+str(gid)+'.pkl'
                    with open(filesavepic, "wb") as ffp:   # pickling
                        pickle.dump(meta_dict,ffp)
                    print("Meta dict", meta_dict)
                simulate_single(df_tmy=df_tmy, 
                                meta_dict=meta_dict, gid=gid,
                                setup=setup, 
                                startdate=startdate, 
                                rootPath=rootPath)

    res = 'FINISHED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    return res

if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run bifacial radiance comparison simulation')
    parser.add_argument('--gid', type=int, required=True,
                        help='GID value to process (single integer)')
    parser.add_argument('--setup', type=int, required=True,
                        help='Setup value to process (single integer)')
    parser.add_argument('--full_year', action='store_true',
                        help='Run full year simulation (default: False, runs 2 days)')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to directory where results will be saved')
    args = parser.parse_args()

    print(">>>>>>>>>>>>>>>>>> STARTING HERE !")
    print(datetime.datetime.now())
    sim_start_time=datetime.datetime.now()

    # Define inputs
    gids = [args.gid]
    setups = [args.setup]
    FullYear = args.full_year

    # dates need to be coerced to 2023 to avoid leap day in while loop
    if FullYear:
        start = datetime.datetime(2023, 1, 1, 0, 0)
        end = datetime.datetime(2023, 12, 31, 0, 0)
    else:
        start = datetime.datetime(2023, 1, 1, 0, 0)
        end = datetime.datetime(2023, 1, 2, 0, 0)

    daylist = []
    while start <= end:
        daylist.append(start)
        start += datetime.timedelta(days=1)

    now=datetime.datetime.now()
    results_path = args.results_path

    print("Bifacial_radiance version ", br.__version__)
    print("Pandas version ", pd.__version__)
    print("Numpy version ", np.__version__)

    starttimer = timer()

    # rootPath = os.getcwd()
    nsrdb_file = '/kfs2/datasets/NSRDB/current/nsrdb_tmy-2024.h5'
    #TMY data located on eagle about 900GB

    #Input
    parameters = ['dhi', 'ghi', 'dni', 'air_temperature',  'wind_speed', 'wind_direction', 'surface_albedo', 
                  'dew_point', 'surface_pressure']

    with NSRDBX(nsrdb_file, hsds=False) as f:
        meta = f.meta
   
    print("NSRDB accessed")

    meta_USA = meta[meta['country'] == 'United States']
    meta_USA = meta_USA.loc[gids]

    data = []
    with NSRDBX(nsrdb_file, hsds=False) as f:
        for p in parameters:
            data.append(f.get_gid_df(p, np.array(list(gids)))) #.values 

    print("GIDs appended")

    #Create multi-level dataframe
    columns = pd.MultiIndex.from_product([parameters, np.array(list(gids))], names=["par", "gid"])
    df_weather = pd.concat(data, axis=1)
    df_weather.columns = columns
    df_weather = df_weather.swaplevel(axis=1).sort_index(axis=1)

    # Pass variables being looped on, and kwargs
    run_simulations(df_weather = df_weather, meta = meta_USA, 
                            setups = setups, startdates = daylist, 
                            rootPath = results_path)

    print("*********** DONE ************")
    stop = timer()
    runTime = round(stop-starttimer,2)

    # =========== Perform Simulation Set ===========
    stop = timer()
    runTime = round(stop-starttimer,2)
    min = int(runTime//60)
    sec = int(round((runTime/60 - min)*60,0))
    print('=======================================')
    print(f'Simulation Run Time: {min:02}:{sec:02}')
    print('=======================================')
    
    with open('TIMER_'+'ALL.pkl', "wb") as tfp:   # Unpickling
        pickle.dump(runTime,tfp)
    #compile(rootPath)

    # Puerto rico (~472 locations), 1 day (13 timestamps) for 5 setups, with about 1/3 of it already modeled
    # took 70 minutes in 10 nodes
    #


