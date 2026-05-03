import bifacial_radiance
 
rad_obj = bifacial_radiance.RadianceObj('makemod', 'TEMP')
 
rad_obj.getEPW(37.42, -110)
 
moduletype='PVmodule'
y = 2
x = 1
 
rad_obj.makeModule(name=moduletype, x=x, y=y)
 
moduletype='PVmodule_1mxgap'
 
rad_obj.makeModule(name=moduletype, x=x, y=y, xgap=1)