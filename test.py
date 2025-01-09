import math
import warp as wp
import warp.sim
import warp.sim.render
import warp.render
import torch

class Example:
    def __init__(self):
        self.substeps = 5
        self.integrator = warp.sim.XPBDIntegrator()
        self.dt = 1/60.0
        self.radius = 0.01
        self.home_q = torch.tensor([0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4, 0.0, 0.0]).cuda()
        self.model = self.build_scene()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.sim_time = 0.0
        self.colors = None
        warp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)
        warp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_1)
        self.sim_renderer = warp.sim.render.SimRendererOpenGL(self.model, path=None)
        joint_act = wp.to_torch(self.control.joint_act)
        joint_act[:] = self.home_q.cuda() # [:] so the original tensor is modified

        with wp.ScopedCapture() as capture:
            self.physics_step_()
        self.graph = capture.graph


    def render(self):
        self.sim_renderer.begin_frame()
        self.sim_renderer.render(self.state_0)
        self.sim_renderer.end_frame()


    def build_scene(self):
        # urdf_path = "assets/cartpole/cartpole.urdf"
        urdf_path = "assets/franka/panda.urdf"
        articulation_builder = warp.sim.ModelBuilder()
        warp.sim.parse_urdf(
            urdf_path,
            articulation_builder,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            density=100,
            armature=0.1,
            damping=0.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False
        )
        # builder = warp.sim.ModelBuilder(up_vector=(0, 0, 1))
        builder = warp.sim.ModelBuilder()
        builder.add_builder(articulation_builder)
        builder.joint_q = self.home_q.tolist()
        model = builder.finalize()
        model.ground=False
        return model


    def physics_step(self, dt):
        wp.capture_launch(self.graph)
        
    
    def physics_step_(self):
        with wp.ScopedTimer("step", print=False):
            for _s in range(self.substeps):
                warp.sim.collide(self.model, self.state_0)
                self.state_0.clear_forces()
                self.state_1.clear_forces()
                self.integrator.simulate(self.model, self.state_0, self.state_1, self.dt/self.substeps, self.control)
                self.state_0, self.state_1 = self.state_1, self.state_0

def main():
    example = Example()
    while example.sim_renderer.is_running():
        example.physics_step(0)
        example.render()

if __name__ == "__main__":
    main()
