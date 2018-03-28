classdef GDparams
    %GDPARAMS Hold parameters controlling learning algorithm
    properties
        n_batch
        eta
        n_epochs
    end
    methods
      function obj = GDparams(x,y,z)
         obj.n_batch = x;
         obj.eta = y;
         obj.n_epochs = z;
      end
   end
end
