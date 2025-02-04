from torch.utils.tensorboard import SummaryWriter

# Initialize SummaryWriter
writer = SummaryWriter(log_dir='./runs')
num_iterations=1000
# Example training loop
for it in range(num_iterations):
    # Assume these are the values you want to log
    mean_reward = 100*it  # Replace with actual value
    mean_loss = it  # Replace with actual value

    # Log metrics to TensorBoard
    writer.add_scalar('train/episode/rew_total/mean', mean_reward, global_step=it)
    writer.add_scalar('train/episode/loss/mean', mean_loss, global_step=it)

    # Other training code here...

# Don't forget to close the writer when done
writer.close()
