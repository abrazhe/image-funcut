_DM_help_msg =  """
Diameter Manager:
- left-click or type 'i' to add point under mouse
- rigth-click or type 'd' to remove point under mouse
- type 'a' to calculate propagation velocity
"""

class GWExpansionMeasurement1:
    def __init__(self, ax):
        self.ax = ax
        self.points = []
        self.line = None # user-picked points
        self.line2 = None
        self.line_par = None
        self._ind = None
        self.epsilon = 5
        self.canvas = ax.figure.canvas
        self.center = None
        self.velocities = None
        self.smooth = 1
        #print self.ax, self.ax.figure, self.canvas
        cf = self.canvas.mpl_connect
        self.cid = {
            'press': cf('button_press_event', self._on_button_press),
            #'release': cf('button_release_event', self.on_release),
            #'motion': cf('motion_notify_event', self.on_motion),
            #'scroll': cf('scroll_event', self.on_scroll),
            'type': cf('key_press_event', self._on_key_press)
            }
        plt.show()
        print(_DM_help_msg)

    def disconnect(self):
        if self.line_par:
            self.line_par.remove()
        if self.line:
            self.line.remove()
        if self.line2:
            self.line2.remove()
        for cc in self.cid.values():
            self.canvas.mpl_disconnect(cc)
        self.canvas.draw()

    def get_ind_closest(self,event):
        xy = np.asarray(self.points)
        if self.line is not None:
            xyt = self.line.get_transform().transform(xy)
            xt, yt = xyt[:,0], xyt[:,1]
            d = ((xt-event.x)**2 + (yt-event.y)**2)**0.5
            indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
            ind = indseq[0]
            if d[ind]>=self.epsilon:
                ind = None
        else: ind = None
        return ind



    def add_point(self, event):
        p = event.xdata, event.ydata
        ind = self.get_ind_closest(event)
        if ind is not None:
            print("too close to an existing point")
            return
        self.points.append(p)
        self.points.sort(key = lambda u:u[0])
        xd, yd = rezip(self.points)
        if self.line is None:
            self.line = plt.Line2D(xd,yd,marker='o',
                                  ls = '--',
                                  color='r', alpha=0.75,
                                  markerfacecolor='r')
                                  #animated='True')
            self.ax.add_line(self.line)
            #print 'added line'
        else:
            self.line.set_data([xd,yd])

    def remove_point(self, event):
        ind = self.get_ind_closest(event)
        if ind is not None:
            self.points = [pj for j,pj in enumerate(self.points) if j !=ind]
            self.line.set_data(rezip(self.points))

    def action(self,min_r = 5.):
        xd, yd = list(map(np.array, rezip(self.points)))
        par = np.polyfit(xd,yd,2)
        xfit_par = np.linspace(xd[0], xd[-1], 256)
        yfit_par = np.polyval(par, xfit_par)
        v = np.gradient(np.asarray(yd))
        dx = np.gradient(np.asarray(xd))
        tck,u = splprep([xd,yd],s=self.smooth)
        unew = np.linspace(0,1.,100)
        out = splev(unew,tck)
        if self.line_par:
            self.line_par.set_data(xfit_par, yfit_par)
        else:
            self.line_par = plt.Line2D(xfit_par, yfit_par, color='cyan')
            self.ax.add_line(self.line_par)
        if self.line2:
            self.line2.set_data(out[0], out[1])
        else:
            self.line2 = plt.Line2D(out[0], out[1], color='w',
                                   lw=2,alpha=0.75)
            self.ax.add_line(self.line2)
        x,y = out[0], out[1]
        midpoint = np.argmin(y)
        lh_r = abs(x[:midpoint]-x[midpoint]) #left branch
        rh_r = x[midpoint:]-x[midpoint] # right branch
        #vel = lambda d,t: np.abs(np.gradient(d)/np.gradient(t))
        def vel(d,t): return np.abs(np.gradient(d)/np.gradient(t))
        rh_v = vel(rh_r[rh_r>=min_r],y[midpoint:][rh_r>=5])
        lh_v = vel(lh_r[lh_r>=min_r],y[:midpoint][lh_r>=5])
        rh_r = rh_r[rh_r>=min_r]
        lh_r = lh_r[lh_r>=min_r]
        #v_at_r = lambda rv,vv,r0: vv[np.argmin(np.abs(rv-r0))]
        def v_at_r(rv,vv,r0): return vv[np.argmin(np.abs(rv-r0))]
        #vmean_at_r = lambda r0:\
        #            np.mean([v_at_r(lh_r,lh_v,r0),
        #                     v_at_r(rh_r,rh_v,r0)])
        def vmean_at_r(r0):
            return np.mean([v_at_r(lh_r,lh_v,r0), v_at_r(rh_r,rh_v,r0)])

        ax = plt.figure().add_subplot(111);
        ax.plot(lh_r[lh_r>min_r],lh_v,'b-',lw=2)
        ax.plot(rh_r[rh_r>min_r],rh_v,'g-',lw=2)
        ax.legend(['left-hand branch','right-hand branch'])
        ax.set_xlabel('radius, um'); ax.set_ylabel('velocity, um/s')
        ax.set_title('average velocity at 15 um: %02.2f um/s'%\
                 vmean_at_r(15.))
        ax.grid(True)
        self.canvas.draw()
        print("-------- Velocities ----------")
        print(np.array([(rx, vmean_at_r(rx)) for rx in range(8,22,2)]))
        print("------------------- ----------")
        self.velocities = [[lh_r[::-1], lh_v[::-1]],
                           [rh_r, rh_v]]
        return

    def _on_button_press(self,event):
        if not self.event_ok(event): return
        x,y = event.xdata, event.ydata
        if event.button == 1:
            self.add_point(event)
        elif event.button == 3:
            self.remove_point(event)
        self.canvas.draw()

    def _on_key_press(self, event):
        if not self.event_ok(event): return
        if event.key == 'i':
            self.add_point(event)
        elif event.key == 'd':
            self.remove_point(event)
        elif event.key == 'a':
            self.action()
        self.canvas.draw()
    def event_ok(self, event):
        return event.inaxes == self.ax and \
               self.canvas.toolbar.mode ==''
