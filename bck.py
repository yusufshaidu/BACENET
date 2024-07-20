
 #tensorprod(rij, rij)
            #tensor_rij_rij = tf.einsum('ijk, ijl -> ijkl',rij_unit, rij_unit)
            #compute the Frobinus norm of the tensors
            #tensor_rij_rij_Fnorm = tf.einsum('ijkl, ijlm -> ijkm',tensor_rij_rij, tf.transpose(tensor_rij_rij, (0,1,3,2)))
            #tf trace apply trace on the last two dimensions. This would be control by axis1 and axis2 in numpy
            # nat x neigh x neigh
            #tensor_rij_rij_Fnorm = tf.sqrt(tf.linalg.trace(tensor_rij_rij_Fnorm) + 1e-12)
            #tensor_rij_rij_Fnorm = tf.tile(tensor_rij_rij_Fnorm[:,:,tf.newaxis], [1,1,Nneigh])
            #tensor_rij_rij_Fnorm = tf.reshape(tensor_rij_rij_Fnorm, [nat,-1])
            #tensor_rij_rij_Fnorm = tensor_rij_rij_Fnorm[:,tf.newaxis,:] * lambda1[tf.newaxis,:,tf.newaxis] / 3.0



            #remove diagonal elements from the tensor product: we create ones of same shape as tensor and set the innermost diagonal to zero
            #then sum over the tensor elements. If we want to keep the diagonal, we could jus do
            #tensor_rij_rij = tf.einsum('ijk,ijl->ij',rij_unit, rij_unit)
            #tl = tf.ones_like(tensor_rij_rij)-tf.eye(3)
            #tensor_rij_rij = tf.reduce_sum(tensor_rij_rij * tl, axis=(-2,-1)) #natxNeigh
            #tile to have same dimension as costheta
            #tensor_rij_rij = tf.tile(tensor_rij_rij[:,:,tf.newaxis], [1,1,Nneigh])
            #tensor_rij_rij = tf.reshape(tensor_rij_rij, [nat,-1])
            ##multiply by the learnable parameter lambda1
            #tensor_rij_rij = tensor_rij_rij[:,tf.newaxis,:] * lambda1[tf.newaxis,:,tf.newaxis] / 3.0

            #tensorprod(rij, rik)
            #tensor_rij_rik = tf.einsum('ijk, ilm -> ijlkm',rij_unit, rij_unit)
            #X = tensorprod(rij, rik)
            #S = (X + X.T) / 2 - 1/3 trace(X) Id

            #S = tensor_rij_rik + tf.transpose(tensor_rij_rik, (0,1,2,4,3))

            #S /= 2.0
            #trace(X) = cos_theta_ijk
            #to compute trace(X) * Id, we need to first tile cos_theta_ijk to have (nat,neigh,neigh,3,3) dimensions
            #then the diagonal in the last 2 directions that are 3x3. We achieve this by multiply tf.eye(3)
            #trace_X = tf.tile(cos_theta_ijk[:,:,:,tf.newaxis,tf.newaxis], [1,1,1,3,3])

            #S -= trace_X * tf.eye(3) / 3.0
            #compute the Frobinius norm of the tensors
            #S_Fnorm = tf.einsum('ijklm, ijkmn -> ijkln',S, tf.transpose(S, (0,1,2,4,3)))
            #tf trace apply trace on the last two dimensions. This would be control by axis1 and axis2 in numpy
            # dimension = nat x neigh x neigh
            #tensors_contrib = tf.sqrt(tf.linalg.trace(S_Fnorm) + 1e-12)
            #tensors_contrib = tf.reshape(tensors_contrib, [nat,-1])
            #tensors_contrib  = tensors_contrib[:,tf.newaxis,:] * lambda1[tf.newaxis,:,tf.newaxis] * tf.sqrt(3.0/2.0)

