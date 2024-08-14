# Chat-AI-Service
A Service to write AI Agent Logic.

## System Flow Diagram

![System Flow Diagram](doc/flow.jpg)


## Services

- [Sync-Backend](https://github.com/AI-at-Work/Sync-Backend)
- [Chat-UI](https://github.com/AI-at-Work/Chat-UI)
- [Chat-Backend](https://github.com/AI-at-Work/Chat-Backend)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/AI-at-Work/Chat-AI-Service
   cd Chat-AI-Service
   ```

2. Copy the `.env.sample` to `.env` and configure the environment variables:
   ```bash
   cp .env.sample .env
   ```
   Edit the `.env` file to set your specific configurations and add openai api key.

3. Start the service:
   ```bash
   make proto && docker compose up -d --build
   ```

# To add Nvidia GPU to the Ollama You Need to install NVIDIA Container Toolkit

1. Configure the production repository:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ```
2. Update the packages list from the repository:
   ```bash
   sudo apt-get update
   ```
3. Install the NVIDIA Container Toolkit packages:
   ```bash
      sudo apt-get install -y nvidia-container-toolkit
   ```
4. Configure the container runtime by using the nvidia-ctk command:
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   ```
5. Restart the Docker daemon:
   ```bash
   sudo systemctl restart docker
   ```

## Configuration

Key configuration options in the `.env` file:

- `AI_*`: AI Server configurations

Refer to the `.env.sample` file for a complete list of configuration options.

