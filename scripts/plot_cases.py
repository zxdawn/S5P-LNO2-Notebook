'''
Plot NO2 column with lightning tracer, LNO2 column, and cloud pressure

INPUT:
    S5P_LNO2_production.nc <- [s5p_lno2_product.py](https://github.com/zxdawn/S5P-LNO2/blob/main/main/s5p_lno2_product.py)

OUTPUT:
    Each case has one jpeg image with three columns and <n_orbit> rows.

UPDATE:
    Xin Zhang:
        2023-01-13: Basic version
'''


import os
import gc
import proplot as pplt
import logging
import numpy as np
import xarray as xr
from netCDF4 import Dataset

# Choose the following line for info or debugging:
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)

class s5p_lno2_img:
    '''Plot the tropomi no2 with lightning images'''
    def __init__(self, filename):
        self.filename = filename
        self.ds = Dataset(self.filename)

        # get the group names of Cases
        self.cases = list(sorted(self.ds.groups.keys()))

    def process_data(self, case):
        # get the group names of swath inside Case
        self.swaths = list(sorted(self.ds[case].groups.keys()))

        # for calculating plotting boundary later
        lons, lats = [], []

        for swath in self.swaths:
            logging.info(f'Processing {swath}')

            # load data
            ds_s5p = xr.open_mfdataset(self.filename, group=case+'/'+swath+'/S5P')
            # ds_lightning = xr.open_mfdataset(filename, group=case+'/'+swath+'/Lightning')

            ds_s5p['longitude'].load()
            ds_s5p['latitude'].load()
            # ds_s5p['lightning_mask'].load()

            # lon = ds_s5p['longitude'].where(ds_s5p['lightning_mask']>0)
            # latitude = ds_s5p['latitude'].where(ds_s5p['lightning_mask']>0)
            lons.extend([ds_s5p['longitude'].min().item(), ds_s5p['longitude'].max().item()])
            lats.extend([ds_s5p['latitude'].min().item(), ds_s5p['latitude'].max().item()])

            if np.all(ds_s5p['nitrogendioxide_tropospheric_column'].isnull()):
                self.swaths.remove(swath)
        
        lons, lats = np.array(lons), np.array(lats)
        lon_min = lons.min()
        lon_max = lons.max()
        lat_min = lats.min()
        lat_max = lats.max()

        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max

    def plot_no2(self, ds_s5p, ds_lightning, axs):
        """Plot the NO2 and lightning NO2 tracer at three pressure levels"""
        cmaps = ['Greens2_r', 'Oranges2_r', 'Blues2_r']

        # plot NO2 VCD
        cbar_kwargs={'label': '($\mu$mol m$^{-2}$)'}

        (self.no2*1e6).plot(ax=axs[0], x='longitude', y='latitude',
                        cmap='Thermal', vmin=0, vmax=30, discrete=False,
                        add_colorbar=True, rasterized=True,
                        cbar_kwargs=cbar_kwargs,#, 'orientation': 'horizontal'},
                        )

        (self.no2*1e6).plot(ax=axs[1], x='longitude', y='latitude',
                        cmap='Thermal', vmin=0, vmax=30, discrete=False,
                        add_colorbar=True, rasterized=True,
                        cbar_kwargs=cbar_kwargs,#, 'orientation': 'horizontal'},
                        )

        ds_lightning = ds_lightning.where(ds_lightning['lightning_label']==self.lightning_label, drop=True)

        ax = axs[1]
        for index,level in enumerate(ds_lightning.level):
            # plot lightning
            s_pred = ax.scatter(ds_lightning['longitude_pred'].sel(level=level),
                                ds_lightning['latitude_pred'].sel(level=level),
                                marker="$\u25EF$", cmap=cmaps[index],
                                c=ds_lightning['delta'], s=3,
                                label=str(level.values)+' hPa')

        leg = ax.legend(loc='t', frame=False, markerscale=4)
        leg.legendHandles[0].set_color('green6')
        leg.legendHandles[1].set_color('orange6')
        leg.legendHandles[2].set_color('blue6')

        axs[0].format(xlabel='', ylabel='', title='')
        axs[1].format(xlabel='', ylabel='', title='')

    def plot_lno2(self, ax):
        """Plot the lightning NO2"""
        # plot LNO2 VCD
        cbar_kwargs={'label': '($\mu$mol m$^{-2}$)'}

        (self.lno2*1e6).plot(ax=ax, x='longitude', y='latitude',
                        cmap='Thermal', vmin=0, vmax=30, discrete=False,
                        add_colorbar=True, rasterized=True,
                        cbar_kwargs=cbar_kwargs,#, 'orientation': 'horizontal'},
                        )
        ax.format(title='')

    def plot_cp(self, ax):
        """Plot cloud pressure"""
        cbar_kwargs={'label': '(hPa)'}

        (self.cp).plot(x='longitude', y='latitude',
                       vmin=150, vmax=700,
                       cmap='Blues',
                       ax=ax,
                       discrete=False,
                       add_colorbar=True, rasterized=True,
                       cbar_kwargs=cbar_kwargs,
                       )
        ax.format(title='')

    def plot_data(self, case):
        fig, axs = pplt.subplots(nrows=len(self.swaths), ncols=4, spanx=0, spany=0)

        filenames = []
        for index,swath in enumerate(self.swaths):
            logging.info(f'Processing {swath}')

            # load data
            ds_s5p = xr.open_mfdataset(self.filename, group=case+'/'+swath+'/S5P')
            ds_lightning = xr.open_mfdataset(self.filename, group=case+'/'+swath+'/Lightning')

            self.lon = ds_s5p['nitrogendioxide_tropospheric_column'].coords['longitude']
            self.lat = ds_s5p['nitrogendioxide_tropospheric_column'].coords['latitude']

            lightning_label = np.unique(ds_s5p['lightning_mask'].values)
            filenames.append(ds_s5p.attrs['s5p_filename'])

            self.no2 = ds_s5p['nitrogendioxide_tropospheric_column']
            self.lno2 = ds_s5p['lno2']
            self.lightning_label = lightning_label[~np.isnan(lightning_label)]
            self.cp = ds_s5p['cloud_pressure_crb']/1e2

            self.plot_no2(ds_s5p, ds_lightning, [axs[index*4], axs[index*4+1]])
            self.plot_lno2(axs[index*4+2])
            self.plot_cp(axs[index*4+3])

            del ds_s5p, ds_lightning
            gc.collect()
        
        # annotation = '\n'.join(['S5P input files:']+filenames)
        annotation = '\n'.join(filenames)
        fig.suptitle(annotation, weight='normal')

        axs.format(xlim=(self.lon_min, self.lon_max), ylim=(self.lat_min, self.lat_max),
                   toplabels=['NO$_2$ Tropospheric Column', 'NO$_2$ Tropospheric Column \n with lightning tracer',
                              'Lightning NO$_2$ Tropospheric Column', 'Cloud Pressure'],
                   grid=False, xformatter='deglon', yformatter='deglat', facecolor='gray5',
                   xlabel='', ylabel='', leftlabels=[s.replace('Swath', 'Orbit') for s in self.swaths],
                   leftlabelrotation='horizontal')
        
        return fig


def main():
    data = s5p_lno2_img(filename)

    for index in range(len(data.cases)):
        # get case number
        case = data.cases[index]

        # process data and plot
        data.process_data(case)
        fig = data.plot_data(case)

        # create dirs
        case_name = f"case{index:02}"
        savedir = '../figures/cases/'
        os.makedirs(savedir, exist_ok=True)

        # save plot
        savename = savedir+case_name+'.jpg'
        logging.info(f'Saving to {savename}')
        fig.savefig(savename, dpi=300)

        del fig
        gc.collect()


if __name__ == '__main__':
    filename = '../data/lno2/S5P_LNO2_production.nc'
    main()