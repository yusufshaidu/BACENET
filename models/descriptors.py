import tensorflow as tf
import functions.helping_functions as help_fn

def cosine_terms(zeta):
    lxlylz, lxlylz_sum, fact_norm = help_fn._compute_cosine_terms(zeta)
    lxlylz = tf.cast(lxlylz,tf.float32) #[n_lxlylz, 3]
    lxlylz_sum = tf.cast(lxlylz_sum, tf.int32) #[n_lxlylz,]
    fact_norm = tf.cast(fact_norm, tf.float32) #[n_lxlylz,]
    return lxlylz, lxlylz_sum, fact_norm

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            ]
             )
def _angular_terms(rij_unit, lxlylz):
    '''
    Compute vectorized three-body angular terms.
    '''
    # Compute powers: shape = [npairs, n_lxlylz, 3]
    rij_lxlylz = (tf.expand_dims(rij_unit, axis=1) + 1e-12) ** tf.expand_dims(lxlylz, axis=0)
    # Multiply x^lx * y^ly * z^lz
    g_ij_lxlylz = tf.reduce_prod(rij_lxlylz, axis=-1)              # [npairs, n_lxlylz]
    #g_ij_lxlylz = rij_lxlylz[:,:,0] * rij_lxlylz[:,:,1] * rij_lxlylz[:,:,2]  
    return g_ij_lxlylz

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            ]
             )
def to_three_body_order_terms_chunk(
    rij_unit,
    radial_ij,
    first_atom_idx,
    nat,
):
    block_size=4
    zeta = 3
    lxlylz, lxlylz_sum, fact_norm = cosine_terms(zeta)


    r_start = 1
    r_end = r_start + 1 + zeta

    nfeat = tf.shape(radial_ij)[1]
    #nfeat = tf.cast(nfeat, tf.int32)
    nzeta = zeta + 1

    ta = tf.TensorArray(
        dtype=tf.float32,
        size=0,
        dynamic_size=True,
        element_shape=(None, nzeta, None),
    )

    g_ang = _angular_terms(rij_unit, lxlylz)  # computed ONCE

    def cond(f0, *_):
        return f0 < nfeat

    def body(f0, ta):
        f1 = tf.minimum(f0 + block_size, nfeat)
        #f1 = tf.cast(f1, tf.int32)

        # radial gather: [npairs, f_block, n_l]
        rad = tf.gather(
            radial_ij[:, f0:f1, r_start:r_end],
            tf.cast(lxlylz_sum,tf.int32),
            axis=2
        )

        g_pair = rad * g_ang[:, None, :]

        g_atom = tf.math.unsorted_segment_sum(
            g_pair,
            first_atom_idx,
            num_segments=nat
        )

        tmp = tf.transpose(g_atom * g_atom, [2, 0, 1]) * fact_norm[:, None, None]

        out = tf.math.unsorted_segment_sum(
            tmp,
            lxlylz_sum,
            num_segments=nzeta
        )

        out = tf.transpose(out, [1, 0, 2])  # [nat, nzeta, f_block]

        ta = ta.write(ta.size(), out)
        #ta = tf.concat([ta, out], axis=2)
        return f1, ta

    _, ta = tf.while_loop(
        cond,
        body,
        loop_vars=(0, ta),
        parallel_iterations=1,
        maximum_iterations=None,
    )

    gi3 = tf.transpose(tf.concat(ta.stack(), axis=2),
                       [1,2,0,3])# [nat, nzeta, nblock, nchuck]
    return tf.reshape(gi3, [nat, -1])

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            ]
             )
def to_three_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat):
    '''
    compute  up to four-body computation

    '''
    zeta = 3
    lxlylz, lxlylz_sum, fact_norm = cosine_terms(zeta)    
    g_ij_lxlylz = _angular_terms(rij_unit,lxlylz) #npairs,n_lxlylz
    shapes = tf.shape(g_ij_lxlylz)
    npairs = shapes[0]
    n_lxlylz = shapes[1]
    
    r_start = 1
    r_end = r_start + 1 + zeta
    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum, axis=2)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = radial_ij_expanded * tf.expand_dims(g_ij_lxlylz, axis=1)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = tf.math.unsorted_segment_sum(g_ilxlylz, first_atom_idx,num_segments=nat)
    # shape: [nat, nspec * nrad, n_lxlylz]

    # Multiply g_ilxlylz^2 and transpose
    _gi3 = tf.transpose(g_ilxlylz * g_ilxlylz, [2,0,1]) * fact_norm[:,None,None]
    # shape: [n_lxlylz, nat, nspec * nrad]

    gi3 = tf.math.unsorted_segment_sum(_gi3, lxlylz_sum, num_segments=(1+zeta))
    # shape: [nzeta, nat, nspec * nrad]
    gi3 = tf.transpose(gi3, perm=(1,0,2)) #nat,nzeta,nrad*nspec
    gi3 = tf.reshape(gi3, [nat, -1])
    return gi3

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            ]
             )
def to_four_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat):
    '''
    compute  up to four-body computation

    '''
    zeta = 3
    lxlylz, lxlylz_sum, fact_norm = cosine_terms(zeta)    
    g_ij_lxlylz = _angular_terms(rij_unit,lxlylz)
    shapes = tf.shape(g_ij_lxlylz)
    npairs = shapes[0]
    n_lxlylz = shapes[1]
    # shape: [npair, n_lxlylz]
    r_start = 1
    r_end = r_start + 1 + zeta
    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum, axis=2)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = radial_ij_expanded * tf.expand_dims(g_ij_lxlylz, axis=1)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = tf.math.unsorted_segment_sum(g_ilxlylz, first_atom_idx,num_segments=nat)
    # shape: [nat, nspec * nrad, n_lxlylz]

    # Multiply g_ilxlylz^2 and transpose
    _gi3 = tf.transpose(g_ilxlylz * g_ilxlylz, [2,0,1]) * fact_norm[:,None,None]
    # shape: [n_lxlylz, nat, nspec * nrad]

    gi3 = tf.math.unsorted_segment_sum(_gi3, lxlylz_sum, num_segments=(1+zeta))
    # shape: [nzeta, nat, nspec * nrad]
    gi3 = tf.transpose(gi3, perm=(1,0,2)) #nat,nzeta,nrad*nspec
    gi3 = tf.reshape(gi3, [nat, -1])

    ########
    zeta = 4
    lxlylz, lxlylz_sum, fact_norm = cosine_terms(zeta)    

    g_ij_lxlylz = _angular_terms(rij_unit,lxlylz)
    shapes = tf.shape(g_ij_lxlylz)
    npairs = shapes[0]
    n_lxlylz = shapes[1]
    # shape: [npair, n_lxlylz]
    r_start = r_end
    r_end = r_start + 1 + zeta
    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum, axis=2)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = radial_ij_expanded * tf.expand_dims(g_ij_lxlylz, axis=1)
    # shape: [npair, nspec * nrad, n_lxlylz]
    g_ilxlylz = tf.math.unsorted_segment_sum(g_ilxlylz, first_atom_idx,num_segments=nat)

    g_i_l1l2 = tf.expand_dims(g_ilxlylz,-1) * tf.expand_dims(g_ilxlylz,-2)
    g_i_l1l2 = tf.reshape(g_i_l1l2, [nat, -1, n_lxlylz*n_lxlylz]) #nat, nrad*nspec,n_lxlylz*n_lxlylz

    #rad_ij contains 2*zata + 1 radial functions
    
    r_end = r_start + 2 * zeta + 1

    lxlylz_sum2 = tf.reshape(lxlylz_sum[None,:] + lxlylz_sum[:,None], [-1])
    fact_norm2 = tf.reshape(fact_norm[None,:] * fact_norm[:, None], [-1])
    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum2, axis=2) # npair, nspec*nrad, n_lxlylz * n_lxlylz

    g_ij_l1_plus_l2 = tf.expand_dims(g_ij_lxlylz,-1) * tf.expand_dims(g_ij_lxlylz,-2) # npair,n_lxlylz,n_lxlylz
    g_ij_l1_plus_l2 = tf.reshape(g_ij_l1_plus_l2, [-1, n_lxlylz*n_lxlylz])

    g_ij_ll = radial_ij_expanded * tf.expand_dims(g_ij_l1_plus_l2, 1)

    #contribution after summing over j
    g_i_l1_plus_l2 = tf.math.unsorted_segment_sum(data=g_ij_ll,
                                    segment_ids=first_atom_idx,num_segments=nat)#nat x nrad*nspec,n_lxlylz,n_lxlylz
    
    g_i_l1l2_ijk = tf.transpose(g_i_l1l2 * g_i_l1_plus_l2, [2,0,1]) * fact_norm2[:,None,None] #n_lxlylz * n_lxlylz, nat, nrad*nspec

    nzeta2 = (1 + zeta) * (1 + zeta)

    g_i_l1l2 = tf.math.unsorted_segment_sum(data=g_i_l1l2_ijk,
                                    segment_ids=lxlylz_sum2, num_segments=nzeta2) # nzeta2, nat, nrad*nspec
    g_i_l1l2 = tf.transpose(g_i_l1l2, perm=[1,0,2])

    gi4 = tf.reshape(g_i_l1l2, [nat, -1])
    return (gi3,gi4)
   
@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            ]
             )
def to_five_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat):
    '''
    compute  up to four-body computation

    '''
    zeta = 3
    lxlylz, lxlylz_sum, fact_norm = cosine_terms(zeta)    
    g_ij_lxlylz = _angular_terms(rij_unit,lxlylz)
    shapes = tf.shape(g_ij_lxlylz)
    npairs = shapes[0]
    n_lxlylz = shapes[1]
    # shape: [npair, n_lxlylz]
    r_start = 1
    r_end = r_start + 1 + zeta
    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum, axis=2)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = radial_ij_expanded * tf.expand_dims(g_ij_lxlylz, axis=1)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = tf.math.unsorted_segment_sum(g_ilxlylz, first_atom_idx,num_segments=nat)
    # shape: [nat, nspec * nrad, n_lxlylz]

    # Multiply g_ilxlylz^2 and transpose
    _gi3 = tf.transpose(g_ilxlylz * g_ilxlylz, [2,0,1]) * fact_norm[:,None,None]
    # shape: [n_lxlylz, nat, nspec * nrad]

    gi3 = tf.math.unsorted_segment_sum(_gi3, lxlylz_sum, num_segments=(1+zeta))
    # shape: [nzeta, nat, nspec * nrad]
    gi3 = tf.transpose(gi3, perm=(1,0,2)) #nat,nzeta,nrad*nspec
    gi3 = tf.reshape(gi3, [nat, -1])
    ########
    zeta = 4
    lxlylz, lxlylz_sum, fact_norm = cosine_terms(zeta)    

    g_ij_lxlylz = _angular_terms(rij_unit,lxlylz)
    shapes = tf.shape(g_ij_lxlylz)
    npairs = shapes[0]
    n_lxlylz = shapes[1]
    # shape: [npair, n_lxlylz]
    r_start = r_end
    r_end = r_start + 1 + zeta
    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum, axis=2)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = radial_ij_expanded * tf.expand_dims(g_ij_lxlylz, axis=1)
    # shape: [npair, nspec * nrad, n_lxlylz]
    g_ilxlylz = tf.math.unsorted_segment_sum(g_ilxlylz, first_atom_idx,num_segments=nat)

    g_i_l1l2 = tf.expand_dims(g_ilxlylz,-1) * tf.expand_dims(g_ilxlylz,-2)
    g_i_l1l2 = tf.reshape(g_i_l1l2, [nat, -1, n_lxlylz*n_lxlylz]) #nat, nrad*nspec,n_lxlylz*n_lxlylz

    #rad_ij contains 2*zata + 1 radial functions
    
    r_end = r_start + 2 * zeta + 1

    lxlylz_sum2 = tf.reshape(lxlylz_sum[None,:] + lxlylz_sum[:,None], [-1])
    fact_norm2 = tf.reshape(fact_norm[None,:] * fact_norm[:, None], [-1])
    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum2, axis=2) # npair, nspec*nrad, n_lxlylz * n_lxlylz

    g_ij_l1_plus_l2 = tf.expand_dims(g_ij_lxlylz,-1) * tf.expand_dims(g_ij_lxlylz,-2) # npair,n_lxlylz,n_lxlylz
    g_ij_l1_plus_l2 = tf.reshape(g_ij_l1_plus_l2, [-1, n_lxlylz*n_lxlylz])

    g_ij_ll = radial_ij_expanded * tf.expand_dims(g_ij_l1_plus_l2, 1)

    #contribution after summing over j
    g_i_l1_plus_l2 = tf.math.unsorted_segment_sum(data=g_ij_ll,
                                    segment_ids=first_atom_idx,num_segments=nat)#nat x nrad*nspec,n_lxlylz,n_lxlylz
    
    g_i_l1l2_ijk = tf.transpose(g_i_l1l2 * g_i_l1_plus_l2, [2,0,1]) * fact_norm2[:,None,None] #n_lxlylz * n_lxlylz, nat, nrad*nspec

    nzeta2 = (1 + zeta) * (1 + zeta)

    g_i_l1l2 = tf.math.unsorted_segment_sum(data=g_i_l1l2_ijk,
                                    segment_ids=lxlylz_sum2, num_segments=nzeta2) # nzeta2, nat, nrad*nspec
    g_i_l1l2 = tf.transpose(g_i_l1l2, perm=[1,0,2])

    gi4 = tf.reshape(g_i_l1l2, [nat, -1])

    zeta = 5
    lxlylz, lxlylz_sum, fact_norm = cosine_terms(zeta)    
    g_ij_lxlylz = _angular_terms(rij_unit,lxlylz)
    shapes = tf.shape(g_ij_lxlylz)
    npairs = shapes[0]
    n_lxlylz = shapes[1]
    
    r_start = r_end
    r_end = r_start + zeta + 1

    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum, axis=2)
    # shape: [npair, nspec * nrad, n_lxlylz]

    g_ilxlylz = radial_ij_expanded * tf.expand_dims(g_ij_lxlylz, axis=1)
    # shape: [npair, nspec * nrad, n_lxlylz]
    g_ilxlylz = tf.math.unsorted_segment_sum(g_ilxlylz, first_atom_idx,num_segments=nat)

    g_i_l1l2l3 = g_ilxlylz[:,:,:,None,None] * g_ilxlylz[:,:,None,:,None] * g_ilxlylz[:,:,None,None,:]

    g_i_l1l2l3 = tf.reshape(g_i_l1l2l3, [nat, -1, n_lxlylz*n_lxlylz*n_lxlylz]) #nat, nrad*nspec,n_lxlylz*n_lxlylz*n_lxlylz

    r_end = r_start + 3 * zeta + 1

    lxlylz_sum3 = tf.reshape(lxlylz_sum[:,None,None] + 
                             lxlylz_sum[None,:,None] + 
                             lxlylz_sum[None,None,:], [-1])

    fact_norm3 = tf.reshape(fact_norm[:, None,None] * fact_norm[None,:,None] * fact_norm[None,None,:], [-1])
    radial_ij_expanded = tf.gather(radial_ij[:,:,r_start:r_end], lxlylz_sum3, axis=2) # npair, nspec*nrad, n_lxlylz * n_lxlylz*n_lxlylz

    g_ij_l123 = g_ij_lxlylz[:,:,None,None] * g_ij_lxlylz[:,None,:,None] * g_ij_lxlylz[:,None,None,:]# npair,n_lxlylz,n_lxlylz,n_lxlylz
    g_ij_l123 = tf.reshape(g_ij_l123, [-1, n_lxlylz*n_lxlylz*n_lxlylz]) #

    g_ij_lll = radial_ij_expanded * tf.expand_dims(g_ij_l123, 1)

    #contribution after summing over j
    g_i_l123 = tf.math.unsorted_segment_sum(data=g_ij_lll,
                                    segment_ids=first_atom_idx,num_segments=nat)#nat x nrad*nspec,n_lxlylz**3

    g_i_l1l2l3_ijk = tf.transpose(g_i_l1l2l3 * g_i_l123, [2,0,1]) * fact_norm3[:,None,None] #n_lxlylz**3, nat, nrad*nspec
    _zeta5 = zeta + 1
    nzeta3 = _zeta5 * _zeta5 * _zeta5
    g_i_l1l2l3 = tf.math.unsorted_segment_sum(data=g_i_l1l2l3_ijk,
                                    segment_ids=lxlylz_sum3, num_segments=nzeta3) # nzeta3, nat, nrad*nspec
    g_i_l1l2l3 = tf.transpose(g_i_l1l2l3, perm=[1,0,2])

    gi5 = tf.reshape(g_i_l1l2l3, [nat, -1])
    return [gi3,gi4,gi5] # including 5 body order

