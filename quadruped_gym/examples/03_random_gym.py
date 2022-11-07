from quadruped_gym.gym.a1_pybullet.env import A1BulletGymEnv

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt 
    
    env = A1BulletGymEnv()
    obs = env.reset()

    for i in range(100):
        print(f"Iteration: {i}")
        obs = env.step(np.random.normal(size=(12,)))
        img = env.render()
        plt.imshow(img)
        plt.savefig(f'images/image_{i}.png')