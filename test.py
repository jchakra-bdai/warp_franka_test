import time
import warp as wp
import warp.sim
import warp.sim.render
import warp.render


class Example:
    def __init__(self):
        self.substeps = 4
        self.integrator = warp.sim.XPBDIntegrator()
        self.dt = 1/60.0
        self.radius = 0.01
        self.model = self.build_scene()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.sim_time = 0.0
        self.colors = None
        self.sim_renderer = warp.sim.render.SimRendererOpenGL(self.model, path=None)

        with wp.ScopedCapture() as capture:
            self.physics_step_()
        self.graph = capture.graph


    def render(self):
        self.sim_renderer.begin_frame()
        self.sim_renderer.render(self.state_0)
        self.sim_renderer.end_frame()


    def build_scene(self):
        urdf_path = "assets/franka/panda.urdf"
        articulation_builder = warp.sim.ModelBuilder()
        warp.sim.parse_urdf(
            urdf_path,
            articulation_builder,
        )
        builder = warp.sim.ModelBuilder()
        builder.add_builder(articulation_builder)
        model = builder.finalize()
        model.ground=False
        return model


    def physics_step(self, dt):
        wp.capture_launch(self.graph)
        
    
    def physics_step_(self):
        with wp.ScopedTimer("step"):
            for _s in range(self.substeps):
                warp.sim.collide(self.model, self.state_0)
                self.state_0.clear_forces()
                self.state_1.clear_forces()
                self.integrator.simulate(self.model, self.state_0, self.state_1, self.dt/self.substeps)
                self.state_0, self.state_1 = self.state_1, self.state_0

def main():
    example = Example()
    example.render()
    print("About to run")
    time.sleep(2)
    while example.sim_renderer.is_running():
        example.physics_step_()
        example.render()

if __name__ == "__main__":
    main()
