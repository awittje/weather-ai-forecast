import torch
import torch.nn as nn


# ============================================================
# Training and validating the CNN model
# ============================================================

def training_model(model, training_data, validation_data, epochs):

    # checking the dimensions
    x, y = next(iter(training_data))
    prediction = model(x)

    print("Input:", x.shape)
    print("Target:", y.shape)
    print("Prediction:", prediction.shape)

    ### defining the loss function 
    loss_fn = torch.nn.MSELoss()

    # defining the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
        )

    best_val_loss = float("inf")

    for epoch in range(epochs):

    # ----------------------------
    # Training
    # ----------------------------

        model.train()

        train_loss = 0.0

        for x, y in training_data:

            optimizer.zero_grad()
        
            prediction = model(x)

            loss = loss_fn(
                prediction,
                y
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(training_data)


        # ----------------------------
        # Validation
        # ----------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for x, y in validation_data:

                prediction = model(x)

                loss = loss_fn(
                    prediction,
                    y
                )

                val_loss += loss.item()

        val_loss /= len(validation_data)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train: {train_loss:.5f} | "
            f"Val: {val_loss:.5f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                "../models/best_weather_cnn.pt"
            )