from netCDF4 import Dataset
import matplotlib.pyplot as plt

class eik_geometry():

    def __init__(self,fname, axs=None):

        print('  reading', fname)
        f = Dataset(fname)

        # 1d
        self.theta    = f.variables['theta'][:]
        self.bmag     = f.variables['bmag'] [:]
        self.gradpar  = f.variables['gradpar'][:]
        self.grho     = f.variables['grho'][:]
        self.gds2     = f.variables['gds2'][:] 
        self.gds21    = f.variables['gds21'][:]
        self.gds22    = f.variables['gds22'][:] 
        self.gbdrift  = f.variables['gbdrift'][:] 
        self.gbdrift0 = f.variables['gbdrift0'][:] 
        self.cvdrift  = f.variables['cvdrift'][:] 
        self.cvdrift0 = f.variables['cvdrift0'][:] 
        self.jacob    = f.variables['jacob'][:] 
        self.Rplot    = f.variables['Rplot'][:] 
        self.Zplot    = f.variables['Zplot'][:] 
        self.aplot    = f.variables['aplot'][:] 
        self.Rprime   = f.variables['Rprime'][:] 
        self.Zprime   = f.variables['Zprime'][:] 
        self.aprime   = f.variables['aprime'][:] 

        # 0d
        self.drhodpsi = f.variables['drhodpsi'][:]
        self.kxfac    = f.variables['kxfac'][:]
        self.Rmaj     = f.variables['Rmaj'][:]
        self.q        = f.variables['q'][:]
        self.shat     = f.variables['shat'][:]

        # data
        self.f   = f
        self.axs = axs


    def plot(self):

        axs = self.axs

        axs[0,0].plot( self.theta, self.theta   )
        axs[0,1].plot( self.theta, self.bmag    )
        axs[0,2].plot( self.theta, self.gradpar )
        axs[0,3].plot( self.theta, self.grho    )
        axs[0,4].plot( self.theta, self.gds2    )
        axs[1,0].plot( self.theta, self.gds21   )
        axs[1,1].plot( self.theta, self.gds22   )
        axs[1,2].plot( self.theta, self.gbdrift )
        axs[1,3].plot( self.theta, self.gbdrift0)
        axs[1,4].plot( self.theta, self.cvdrift )
        axs[2,0].plot( self.theta, self.cvdrift0)
        axs[2,1].plot( self.theta, self.jacob   )
        axs[2,2].plot( self.theta, self.Rplot   )
        axs[2,3].plot( self.theta, self.Zplot   )
        axs[2,4].plot( self.theta, self.aplot   )
        axs[3,0].plot( self.theta, self.Rprime  )
        axs[3,1].plot( self.theta, self.Zprime  )
        axs[3,2].plot( self.theta, self.aprime  )

        axs[0,0].set_title( 'theta'   )
        axs[0,1].set_title( 'bmag'    )
        axs[0,2].set_title( 'gradpar' )
        axs[0,3].set_title( 'grho'    )
        axs[0,4].set_title( 'gds2'    )
        axs[1,0].set_title( 'gds21'   )
        axs[1,1].set_title( 'gds22'   )
        axs[1,2].set_title( 'gbdrift' )
        axs[1,3].set_title( 'gbdrift0')
        axs[1,4].set_title( 'cvdrift' )
        axs[2,0].set_title( 'cvdrift0')
        axs[2,1].set_title( 'jacob'   )
        axs[2,2].set_title( 'Rplot'   )
        axs[2,3].set_title( 'Zplot'   )
        axs[2,4].set_title( 'aplot'   )
        axs[3,0].set_title( 'Rprime'  )
        axs[3,1].set_title( 'Zprime'  )
        axs[3,2].set_title( 'aprime'  )


        axs[3,4].plot( self.Rplot, self.Zplot )

class VMEC_Geometry():

    def __init__(self,fname, axs=None):

        print('  reading', fname)
        f = Dataset(fname)

        # 1d
        self.theta    = f.variables['theta'][:]
        self.bmag     = f.variables['bmag'] [:]
        self.gradpar  = f.variables['gradpar'][:]
        self.grho     = f.variables['grho'][:]
        self.gds2     = f.variables['gds2'][:] 
        self.gds21    = f.variables['gds21'][:]
        self.gds22    = f.variables['gds22'][:] 
        self.gbdrift  = f.variables['gbdrift'][:] 
        self.gbdrift0 = f.variables['gbdrift0'][:] 
        self.cvdrift  = f.variables['cvdrift'][:] 
        self.cvdrift0 = f.variables['cvdrift0'][:] 

        # 0d
        self.drhodpsi = f.variables['drhodpsi'][:]
        self.Rmaj     = f.variables['Rmaj'][:]
        self.shat     = f.variables['shat'][:]
        self.kxfac    = f.variables['kxfac'][:]
        self.q        = f.variables['q'][:]
        self.scale    = f.variables['scale'][:]

        # data
        self.f   = f
        self.axs = axs


    def plot(self):

        axs = self.axs

        axs[0,0].plot( self.theta, self.theta   )
        axs[0,1].plot( self.theta, self.bmag    )
        axs[0,2].plot( self.theta, self.gradpar )
        axs[1,0].plot( self.theta, self.grho    )
        axs[1,1].plot( self.theta, self.gds2    )
        axs[1,2].plot( self.theta, self.gds21   )
        axs[1,3].plot( self.theta, self.gds22   )
        axs[2,0].plot( self.theta, self.gbdrift )
        axs[2,1].plot( self.theta, self.gbdrift0)
        axs[2,2].plot( self.theta, self.cvdrift )
        axs[2,3].plot( self.theta, self.cvdrift0)

        axs[0,0].set_title( 'theta'   )
        axs[0,1].set_title( 'bmag'    )
        axs[0,2].set_title( 'gradpar' )
        axs[1,0].set_title( 'grho'    )
        axs[1,1].set_title( 'gds2'    )
        axs[1,2].set_title( 'gds21'   )
        axs[1,3].set_title( 'gds22'   )
        axs[2,0].set_title( 'gbdrift' )
        axs[2,1].set_title( 'gbdrift0')
        axs[2,2].set_title( 'cvdrift' )
        axs[2,3].set_title( 'cvdrift0')



