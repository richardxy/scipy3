""" Plotting routines based on Gist (requires X-server)

  All Gist functions are also available as xplt.<command>

    histogram -- Plot a histogram.
    barplot   -- Construct a barplot.
    errorbars -- Draw connected points with (y-only) errorbars.
    legend    -- Construct and place a legend.
    arrow     -- Draw an arrow.
    plot      -- Plot curves.
    surf      -- Plot surfaces (no axes currently plotted).
    xlabel    -- Place a label on the x-axis.
    ylabel    -- Place a label on the y-axis.
    title     -- Place a title above the plot.
    hold      -- Draw subsequent plots over the current plot.
    matplot   -- Plot many curves in a matrix against a single x-axis.
    addbox    -- Add a box to the current plot.
    imagesc   -- Draw an image.
    imagesc_cb -- Draw an image with a colorbar.
    movie     -- Play a sequence of images as a movie.
    figure    -- Create a new figure.
    full_page -- Create a full_page window.
    subplot   -- Draw a sub-divided full-page plot.
    plotframe -- Change the plot system on a multi-plot page.
    twoplane  -- Create a plot showing two orthogonal planes of a
                 three-dimensional array.
"""

from gist import *
import os, sys
from Mplot import *
from write_style import *
gistpath = os.path.join(sys.prefix, 'lib', 'python%s' % sys.version[:3],
                        'site-packages','scipy','xplt')
os.environ['GISTPATH'] = gistpath

maxwidth=os.environ.get('XPLT_MAXWIDTH')
maxheight=os.environ.get('XPLT_MAXHEIGHT')
if maxwidth is None or maxheight is None:
    import commands
    str = commands.getoutput('xwininfo -root')
    ind1 = str.find('Width:')
    ind2 = str.find('\n',ind1)
    maxwidth=int(str[ind1+6:ind2])-8
    ind1 = str.find('Height:')
    ind2 = str.find('\n',ind1)
    maxheight=int(str[ind1+7:ind2])-60
    os.environ['XPLT_MAXWIDTH']=maxwidth
    os.environ['XPLT_MAXHEIGHT']=maxheight
    

