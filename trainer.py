import torch

# Third-party packages

def trainer(model, optimizer, data_loader, device, lr_scheduler, loss_func,
                    swa_model, swa_scheduler):
    
    model.train()

    epoch_loss = 0
    for image, target in data_loader:
        image, target = image.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
        
        output = model(image)
        loss = loss_func(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        #swa_model.update_parameters(model)
        #swa_scheduler.step()

        epoch_loss += loss.item() * image.size(0)
        
    lr = optimizer.param_groups[0]["lr"]
    lr_scheduler.step()

    return epoch_loss / len(data_loader.dataset), lr