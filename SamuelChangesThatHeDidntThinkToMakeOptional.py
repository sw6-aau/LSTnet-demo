# Iterates through all the batches as inputs.
        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = self.output(model(X.float()));
            if predict is None:
                predict = output;
                test = X;
            else:
                predict = torch.cat((predict,output));
                test = torch.cat((test, X));
            
            # Loss calculation
            scale = data.scale.expand(output.size(0), 168, data.m)
            total_loss += evaluateL2(output * scale, X * scale).data
            total_loss_l1 += evaluateL1(output * scale, X * scale).data
            n_samples += (output.size(0) * data.m);
        
        rse = math.sqrt(total_loss / n_samples)/data.rse
        rae = (total_loss_l1/n_samples)/data.rae