import torch


def train(model, optimizer, loss_function, train_generator, test_generator, epochs, writer=None, loss_step=None,
          transform_batch=None):

    total_batches = 0
    for epoch in range(epochs):
        train_loss = 0.0
        processed_batches = 0
        for i, batch in enumerate(train_generator):
            processed_batches += 1

            batch_x, batch_y = batch
            if transform_batch is not None:
                batch_x = transform_batch(batch_x)

            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_function(batch_y, prediction)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if loss_step is not None and (total_batches + processed_batches) % loss_step == 0:
                if writer is not None:
                    writer.add_scalar('metric/train_step_loss', train_loss / processed_batches, total_batches + processed_batches)

        total_batches += processed_batches

        test_loss = 0.0
        processed_test_batches = 0
        with torch.no_grad():
            for i, batch in enumerate(test_generator):
                processed_test_batches += 1
                batch_x, batch_y = batch
                if transform_batch is not None:
                    batch_x = transform_batch(batch_x)

                prediction = model(batch_x)
                loss = loss_function(batch_y, prediction)

                test_loss += loss.item()

        if writer is not None:
            writer.add_scalars('metric/average_loss', {
                'train': train_loss / processed_batches,
                'test': test_loss / processed_test_batches
            }, epoch)

