from provided_code import data_loaders as dl

files = ["data/balance-scale/balance-scale.data",
         "data/balloons/balloons.data",
         "data/chess-krvk/chess-krvk.data",
         "data/chess-krvkp/chess-krvkp.data",
         "data/connect-4/connect-4.data",
         "data/contraceptive-method/contraceptive-method.data",
         "data/habermans-survival/habermans-survival.data",
         "data/hayes-roth/hayes-roth.data",
         "data/led-display/led-display.data",
         "data/lymphography/lymphography.data",
         "data/molecular-promoters/molecular-promoters.data",
         "data/molecular-splice/molecular-splice.data",
         "data/monks-1/monks-1.data",
         "data/monks-2/monks-2.data",
         "data/monks-3/monks-3.data",
         "data/nursery/nursery.data",
         "data/optdigits/optdigits.data",
         "data/pendigits/pendigits.data",
         "data/semeion/semeion.data",
         "data/spect-heart/spect-heart.data",
         "data/tic-tac-toe/tic-tac-toe.data",
         "data/zoo/zoo.data"]
X, y = dl.load_tabular_xy("data/balance-scale/balance-scale.data")
print(X.shape)
print(X.dtype)
print(X.ndim)