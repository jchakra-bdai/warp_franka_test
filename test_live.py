import math
import time
import warp as wp
import warp.sim
import warp.sim.render
import warp.render
import torch
import trio
# from panda_desk import Desk
from panda_py import Panda

class Example:
    def __init__(self, first_joint_state):
        self.substeps = 4
        self.dt = 1/60.0
        self.radius = 0.01
        self.home_q = torch.tensor(first_joint_state).cuda()
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
        joint_act[:7] = self.home_q.cuda() # [:] so the original tensor is modified
        self.integrator = warp.sim.FeatherstoneIntegrator(self.model)

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
        articulation_builder = warp.sim.ModelBuilder(gravity=0.0)
        warp.sim.parse_urdf(
            urdf_path,
            articulation_builder,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            # density=10.0,
            armature=0.01,
            # damping=1.0,
            # ignore_inertial_definitions=True,
            # damping=80.0,
            # stiffness=400.0,
            damping=10.0,
            stiffness=2000.0,

            # limit_ke=1.0e4,
            # limit_kd=1.0e1,
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

async def main():

    host = "10.103.1.111"
    panda= Panda(host)
    # desk = Desk("10.103.1.111", platform="fr3")
    # await desk.login("admin", "Password!")
    # await desk.take_control(force=True)
    # first_joint_state = None
    # async with desk.robot_states() as gen:
    #     async for state in gen:
    #         first_joint_state = torch.tensor(state["jointAngles"]).cuda()
    #         break

    example = Example(first_joint_state=panda.q)

    # async def update_robot():
    #     desk = Desk("10.103.1.111", platform="fr3")
    #     await desk.login("admin", "Password!")
    #     await desk.take_control(force=True)
    #     print("Control taken")
    #     async with desk.robot_states() as gen:
    #         async for state in gen:
    #             joint_act = wp.to_torch(example.control.joint_act)
    #             joint_act[:7] = torch.tensor(state["jointAngles"]).cuda()
    
    async with trio.open_nursery() as nursery:
        # nursery.start_soon(update_robot)
        while example.sim_renderer.is_running():
            joint_act = wp.to_torch(example.control.joint_act)
            q = panda.get_robot().read_once().q
            joint_act[:7] = torch.tensor(q).cuda()
            t = time.time()
            example.physics_step(0)
            example.render()
            await trio.sleep(1/60)
        nursery.cancel_scope.cancel()   

if __name__ == "__main__":
    trio.run(main)
