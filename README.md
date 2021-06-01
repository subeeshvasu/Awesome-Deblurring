<!--A curated list of resources for Image and Video Deblurring-->
<!-- PROJECT LOGO -->
<p align="center">
  <h3 align="center">Image and Video Deblurring</h3>
  <p align="center">A curated list of resources for Image and Video Deblurring
    <br />
    <br />
    <br />
    <a href="https://github.com/subeeshvasu/Awesome-Deblurring/pulls/new">Suggest new item</a>
    <br />
    <a href="https://github.com/subeeshvasu/Awesome-Deblurring/issues/new">Report Bug</a>
  </p>
</p>

## Table of contents

- [Single-Image-Blind-Motion-Deblurring (non-DL)](#single-image-blind-motion-deblurring-non-dl)
- [Single-Image-Blind-Motion-Deblurring (DL)](#single-image-blind-motion-deblurring-dl)
- [Non-Blind-Deblurring](#non-blind-deblurring)
- [(Multi-image/Video)-Motion-Deblurring](#multi-imagevideo-motion-deblurring)
- [Other Closely Related Works](#other-closely-related-works)
- [Defocus Deblurring and Potential Datasets](#defocus-deblurring-and-potential-datasets)
- [Benchmark Datasets on Motion Deblurring](#benchmark-datasets-on-motion-deblurring)

## Single-Image-Blind-Motion-Deblurring (non-DL)
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2006|TOG|[Removing camera shake from a single photograph](https://cs.nyu.edu/~fergus/papers/deblur_fergus.pdf)|[Code & Project page](https://cs.nyu.edu/~fergus/research/deblur.html)|
|2007|CVPR|[Single image motion deblurring using transparency](http://jiaya.me/all_final_papers/motion_deblur_cvpr07.pdf)||
|2008|CVPR|[Psf estimation using sharp edge prediction](http://vision.ucsd.edu/kriegman-grp/research/psf_estimation/psf_estimation.pdf)|[Project page](http://vision.ucsd.edu/kriegman-grp/research/psf_estimation/)|
|2008|TOG|[High-quality motion deblurring from a single image](http://www.cse.cuhk.edu.hk/~leojia/projects/motion_deblurring/deblur_siggraph08.pdf)|[Code & Project page](http://www.cse.cuhk.edu.hk/~leojia/projects/motion_deblurring/index.html)|
|2009|TOG|[Fast motion deblurring](https://vclab.dgist.ac.kr/download/fast_motion_deblurring/paper.pdf)||
|2009|CVPR|[Image deblurring and denoising using color priors](http://neelj.com/projects/twocolordeconvolution/two_color_deconvolution.pdf)|[Project page](http://neelj.com/projects/twocolordeconvolution/)|
|2010|CVPR|[Efficient ̈filter flow for space-variant multiframe blind deconvolution](https://pure.mpg.de/rest/items/item_1789030/component/file_3009627/content)||
|2010|CVPR|[Non-uniform deblurring for shaken images](http://www.di.ens.fr/willow/pdfs/cvpr10d.pdf)|[Code & Project page](https://www.di.ens.fr/willow/research/deblurring/)|
|2010|CVPR|[Denoising vs. deblurring: HDR imaging techniques using moving cameras](https://ieeexplore.ieee.org/document/5540171)||
|2010|ECCV|[Single image deblurring using motion density functions](http://grail.cs.washington.edu/projects/mdf_deblurring/gupta_mdf_deblurring.pdf)|[Project page](http://grail.cs.washington.edu/projects/mdf_deblurring/)|
|2010|ECCV|[Two-phase kernel estimation for robust motion deblurring](http://www.cse.cuhk.edu.hk/~leojia/projects/robust_deblur/robust_motion_deblurring.pdf)|[Code & Project page](http://www.cse.cuhk.edu.hk/~leojia/projects/robust_deblur/index.html)|
|2010|NIPS|[Space-variant single-image blind deconvolution for removing camera shake](https://papers.nips.cc/paper/4007-space-variant-single-image-blind-deconvolution-for-removing-camera-shake.pdf)||
|2011|CVPR|[Blind deconvolution using a normalized sparsity measure](https://dilipkay.files.wordpress.com/2019/04/priors_cvpr11.pdf)|[Code & Project page](https://dilipkay.wordpress.com/blind-deconvolution/)|
|2011|CVPR|[Blur kernel estimation using the radon transform](http://people.csail.mit.edu/sparis/publi/2011/cvpr_radon/Cho_11_Blur_Kernel_Estimation.pdf)|[Code](http://people.csail.mit.edu/taegsang/Thesis.html)|
|2011|CVPR|[Exploring aligned complementary image pair for blind motion deblurring](https://ieeexplore.ieee.org/document/5995351)||
|2011|ICCV|[Fast removal of non-uniform camera shake](http://pixel.kyb.tuebingen.mpg.de/fast_removal_of_camera_shake/files/Hirsch_ICCV2011_Fast%20removal%20of%20non-uniform%20camera%20shake.pdf)||
|2011|IJCV|[The non-parametric sub-pixel local point spread function estimation is a well posed problem](https://link.springer.com/article/10.1007/s11263-011-0460-0)||
|2012|ECCV|[Blur-kernel estimation from spectral irregularities](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.646.4404&rep=rep1&type=pdf)||
|2012|ACCV|[MRF-based Blind Image Deconvolution](http://imagine.enpc.fr/~komodakn/publications/docs/accv2012.pdf)||
|2013|CVPR|[Unnatural L0 sparse representation for natural image deblurring](http://www.cse.cuhk.edu.hk/~leojia/projects/l0deblur/l0deblur_cvpr13.pdf)|[Code & Project page](http://www.cse.cuhk.edu.hk/~leojia/projects/l0deblur/)|
|2013|CVPR|[Handling noise in single image deblurring using directional filters](http://openaccess.thecvf.com/content_cvpr_2013/html/Zhong_Handling_Noise_in_2013_CVPR_paper.html)|
|2013|NIPS|[Non-Uniform Camera Shake Removal Using a Spatially-Adaptive Sparse Penalty](https://papers.nips.cc/paper/4864-non-uniform-camera-shake-removal-using-a-spatially-adaptive-sparse-penalty)|[Project page](https://sites.google.com/site/hczhang1/projects/non-uniform-camera-shake-removal)|
|2013|ICCV|[Dynamic Scene Deblurring](https://cv.snu.ac.kr/publication/conf/2013/DSD_ICCV2013.pdf)||
|2013|ICCP|[Edge-based blur kernel estimation using patch priors](http://cs.brown.edu/~lbsun/deblur2013/patchdeblur_iccp2013.pdf)|[Project page & Results & Dataset](http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html)|
|2014|CVPR|[Deblurring Text Images via L0 -Regularized Intensity and Gradient Prior](https://eng.ucmerced.edu/people/zhu/CVPR14_deblurtext.pdf)|[Code & Project page](https://sites.google.com/site/jspanhomepage/l0rigdeblur)|
|2014|CVPR|[Segmentation-Free Dynamic Scene Deblurring](https://cv.snu.ac.kr/publication/conf/2014/SFDSD_CVPR2014.pdf)||
|2014|CVPR|[Separable Kernel for Image Deblurring](http://openaccess.thecvf.com/content_cvpr_2014/html/Fang_Separable_Kernel_for_2014_CVPR_paper.html)|
|2014|CVPR|[Deblurring Low-light Images with Light Streaks](https://eng.ucmerced.edu/people/zhu/CVPR14_lightstreak.pdf)|[Code & Project page](https://eng.ucmerced.edu/people/zhu/CVPR14_lightstreak.html)|
|2014|CVPR|[Joint depth estimation and camera shake removal from single blurry image](https://eng.ucmerced.edu/people/zhu/CVPR14_deblurdepth.pdf)||
|2014|ECCV|[Hybrid Image Deblurring by Fusing Edge and Power Spectrum Information](http://www.juew.org/publication/ECCV14-hybridDeblur.pdf)||
|2014|ECCV|[Deblurring Face Images with Exemplars](https://faculty.ucmerced.edu/mhyang/papers/eccv14_deblur.pdf)|[Code & Project page](https://eng.ucmerced.edu/people/zhu/ECCV14_facedeblur.html)|
|2014|ECCV|[Blind deblurring using internal patch recurrence](http://www.wisdom.weizmann.ac.il/~vision/BlindDeblur/Michaeli_Irani_ECCV2014.pdf)|[Code & Project page](http://www.wisdom.weizmann.ac.il/~vision/BlindDeblur.html)|
|2014|NIPS|[Scale Adaptive Blind Deblurring](https://papers.nips.cc/paper/5566-scale-adaptive-blind-deblurring)|[Project page](https://sites.google.com/site/hczhang1/projects/scale-adaptive-blind-deblurring)|
|2015|CVPR|[Burst Deblurring: Removing Camera Shake Through Fourier Burst Accumulation](http://dev.ipol.im/~mdelbra/fba/FBA_cvpr2015_preprint.pdf)|[Project page](http://iie.fing.edu.uy/~mdelbra/fba/)|
|2015|CVPR|[Kernel Fusion for Better Image Deblurring](http://openaccess.thecvf.com/content_cvpr_2015/html/Mai_Kernel_Fusion_for_2015_CVPR_paper.html)|[Project page](http://web.cecs.pdx.edu/~fliu/project/kernelfusion/)|
|2015|ICCV|[Class-Specific Image Deblurring](http://openaccess.thecvf.com/content_iccv_2015/html/Anwar_Class-Specific_Image_Deblurring_ICCV_2015_paper.html)|[Code & Project page](https://github.com/saeed-anwar/Class_Specific_Deblurring)|
|2015|TIP|[Coupled Learning for Facial Deblur](https://arxiv.org/pdf/1904.08671.pdf)||
|2016|CVPR|[Blind image deblurring using dark channel prior](http://vllab1.ucmerced.edu/~jinshan/projects/dark-channel-deblur/dark-channel-deblur/cvpr16-dark-channel-deblur.pdf)|[Code & Project page](http://vllab1.ucmerced.edu/~jinshan/projects/dark-channel-deblur/)|
|2016|CVPR|[Robust Kernel Estimation with Outliers Handling for Image Deblurring](http://openaccess.thecvf.com/content_cvpr_2016/html/Pan_Robust_Kernel_Estimation_CVPR_2016_paper.html)|[Code](https://www.dropbox.com/s/hz9qmi8ar1k1zn0/pcode.zip?dl=0)|
|2016|CVPR|[Blind image deconvolution by automatic gradient activation](http://openaccess.thecvf.com/content_cvpr_2016/papers/Gong_Blind_Image_Deconvolution_CVPR_2016_paper.pdf)||
|2017|CVPR|[Image deblurring via extreme channels prior](http://openaccess.thecvf.com/content_cvpr_2017/html/Yan_Image_Deblurring_via_CVPR_2017_paper.html)|[Code & Project page](https://sites.google.com/site/renwenqi888/research/deblurring/ecp)|
|2017|CVPR|[From local to global: Edge profiles to camera motion in blurred images](http://openaccess.thecvf.com/content_cvpr_2017/html/Vasu_From_Local_to_CVPR_2017_paper.html)|[Project page & Results-on-benchmark-datasets](https://subeeshvasu.github.io/2017_subeesh_from_cvpr/)|
|2017|ICCV|[Blind Image Deblurring with Outlier Handling](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dong_Blind_Image_Deblurring_ICCV_2017_paper.pdf)|[Code](https://www.dropbox.com/s/qmxkkwgnmuwrfoj/code_iccv2017_outlier.zip?dl=0)|
|2017|ICCV|[Self-paced Kernel Estimation for Robust Blind Image Deblurring](http://openaccess.thecvf.com/content_ICCV_2017/papers/Gong_Self-Paced_Kernel_Estimation_ICCV_2017_paper.pdf)|[Code](https://donggong1.github.io/publications.html),[Results](https://drive.google.com/open?id=1gP_s-87js7KKFrIzAlushc1HJqEogR1L)|
|2017|ICCV|[Convergence Analysis of MAP based Blur Kernel Estimation](http://openaccess.thecvf.com/content_iccv_2017/html/Cho_Convergence_Analysis_of_ICCV_2017_paper.html)||
|2018|ECCV|[Normalized Blind Deconvolution](http://openaccess.thecvf.com/content_ECCV_2018/html/Meiguang_Jin_Normalized_Blind_Deconvolution_ECCV_2018_paper.html)|[Code](https://github.com/MeiguangJin/NBD)|
|2018|ECCV|[Deblurring Natural Image Using Super-Gaussian Fields](http://openaccess.thecvf.com/content_ECCV_2018/html/Yuhang_Liu_Deblurring_Natural_Image_ECCV_2018_paper.html)|[Code](https://donggong1.github.io/publications.html)|
|2019|CVPR|[Blind Image Deblurring With Local Maximum Gradient Prior](http://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Blind_Image_Deblurring_With_Local_Maximum_Gradient_Prior_CVPR_2019_paper.html)||
|2019|CVPR|[Phase-Only Image Based Kernel Estimation for Single Image Blind Deblurring](http://openaccess.thecvf.com/content_CVPR_2019/html/Pan_Phase-Only_Image_Based_Kernel_Estimation_for_Single_Image_Blind_Deblurring_CVPR_2019_paper.html)|[Results-on-benchmark-datasets](https://github.com/panpanfei/Phase-only-Image-Based-Kernel-Estimation-for-Blind-Motion-Deblurring/tree/master/result)|
|2019|CVPR|[A Variational EM Framework With Adaptive Edge Selection for Blind Motion Deblurring](http://openaccess.thecvf.com/content_CVPR_2019/html/Yang_A_Variational_EM_Framework_With_Adaptive_Edge_Selection_for_Blind_CVPR_2019_paper.html)||
|2019|TIP|[Graph-Based Blind Image Deblurring From a Single Photograph](https://arxiv.org/abs/1802.07929)|[Code](https://github.com/BYchao100/Graph-Based-Blind-Image-Deblurring)|
|2019|TPAMI|[Surface-aware Blind Image Deblurring](https://ieeexplore.ieee.org/document/8839600)||
|2019|TCSVT|[Single Image Blind Deblurring Using Multi-Scale Latent Structure Prior](https://arxiv.org/abs/1906.04442)||
|2020|ECCV|[OID: Outlier Identifying and Discarding in Blind Image Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5134_ECCV_2020_paper.php)||
|2020|ECCV|[Enhanced Sparse Model for Blind Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700630.pdf)||
|2020|Arxiv|[Blind Image Deblurring Using Patch-Wise Minimal Pixels Regularization](https://128.84.21.199/abs/1906.06642v3)|[Code](https://github.com/FWen/deblur-pmp)|
|2020|Arxiv|[Polyblur: Removing mild blur by polynomial reblurring](https://arxiv.org/abs/2012.09322)||
|2021|SPIC|[Fast blind deconvolution using a deeper sparse patch-wise maximum gradient prior](https://www.sciencedirect.com/science/article/pii/S0923596520301910)||


## Single-Image-Blind-Motion-Deblurring (DL)
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2015|CVPR|[Learning a convolutional neural network for non-uniform motion blur removal](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Learning_a_Convolutional_2015_CVPR_paper.pdf)|[Code 1](http://gr.xjtu.edu.cn/c/document_library/get_file?folderId=2076150&name=DLFE-78101.zip),[Code 2](https://github.com/Sibozhu/MotionBlur-detection-by-CNN)|
|2015|BMVC|[Convolutional neural networks for direct text deblurring](http://www.bmva.org/bmvc/2015/papers/paper006/index.html)|[Code and Project Page](http://www.fit.vutbr.cz/~ihradis/CNN-Deblur/)|
|2016|ECCV|[A neural approach to blind motion deblurring](https://arxiv.org/abs/1603.04771)|[Code](https://github.com/ayanc/ndeblur)|
|2016|PAMI|[Learning to deblur](https://arxiv.org/pdf/1406.7444.pdf)||
|2017|CVPR|[Deep multi-scale convolutional neural network for dynamic scene deblurring](http://zpascal.net/cvpr2017/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)|[Code](https://github.com/SeungjunNah/DeepDeblur_release)|
|2017|CVPR|[From Motion Blur to Motion Flow: A Deep Learning Solution for Removing Heterogeneous Motion Blur](http://openaccess.thecvf.com/content_cvpr_2017/papers/Gong_From_Motion_Blur_CVPR_2017_paper.pdf)|[Code & Project page](https://donggong1.github.io/blur2mflow.html)|
|2017|ICCV|[Blur-Invariant Deep Learning for Blind Deblurring](http://openaccess.thecvf.com/content_ICCV_2017/papers/Nimisha_Blur-Invariant_Deep_Learning_ICCV_2017_paper.pdf)||
|2017|ICCV|[Learning to Super-resolve Blurry Face and Text Images](http://faculty.ucmerced.edu/mhyang/papers/iccv2017_gan_super_deblur.pdf)|[Code & Project page](https://sites.google.com/view/xiangyuxu/deblursr_iccv17)|
|2017|ICCV|[Learning Discriminative Data Fitting Functions for Blind Image Deblurring](http://openaccess.thecvf.com/content_ICCV_2017/papers/Pan_Learning_Discriminative_Data_ICCV_2017_paper.pdf)|[Code](https://www.dropbox.com/s/oavk46q521fiowr/iccv17_learning_deblur_code.zip?dl=0)|
|2018|ICIP|[Semi-supervised Learning of Camera Motion from a Blurred Image](https://apvijay.github.io/pdf/2018_icip.pdf)||
|2018|TIP|[Motion blur kernel estimation via deep learning](https://ieeexplore.ieee.org/abstract/document/8039224)|[Code & Project page](https://sites.google.com/view/xiangyuxu/deepedge_tip)|
|2018|CVPR|[Deep Semantic Face Deblurring](http://openaccess.thecvf.com/content_cvpr_2018/html/Shen_Deep_Semantic_Face_CVPR_2018_paper.html)|[Code](https://github.com/joanshen0508/Deep-Semantic-Face-Deblurring)|
|2018|CVPR|[Learning a Discriminative Prior for Blind Image Deblurring](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Learning_a_Discriminative_CVPR_2018_paper.html)|[Code & Project page](https://sites.google.com/view/lerenhanli/homepage/learn_prior_deblur)|
|2018|CVPR|[Dynamic Scene Deblurring Using Spatially Variant Recurrent Neural Networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Dynamic_Scene_Deblurring_CVPR_2018_paper.html)|[Code](https://github.com/zhjwustc/cvpr18_rnn_deblur_matcaffe)|
|2018|CVPR|[Scale-recurrent network for deep image deblurring](http://openaccess.thecvf.com/content_cvpr_2018/html/Tao_Scale-Recurrent_Network_for_CVPR_2018_paper.html)|[Code](https://github.com/jiangsutx/SRN-Deblur)|
|2018|CVPR|[Deblurgan: Blind motion deblurring using conditional adversarial networks](http://openaccess.thecvf.com/content_cvpr_2018/html/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.html)|[Code-Pytorch](https://github.com/KupynOrest/DeblurGAN)|
|2018|ECCV|[Unsupervised Class-Specific Deblurring](http://openaccess.thecvf.com/content_ECCV_2018/html/Nimisha_T_M_Unsupervised_Class-Specific_Deblurring_ECCV_2018_paper.html)||
|2018|BMVC|[Gated Fusion Network for Joint Image Deblurring and Super-Resolution](https://arxiv.org/abs/1807.10806)|[Code](https://github.com/jacquelinelala/GFN)|[Project page](http://xinyizhang.tech/bmvc2018/)|
|2019|WACV|[Gyroscope-Aided Motion Deblurring with Deep Networks](https://arxiv.org/abs/1810.00986)|[Code](https://github.com/jannemus/DeepGyro)|
|2019|CVPR|[Deep Plug-And-Play Super-Resolution for Arbitrary Blur Kernels](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Deep_Plug-And-Play_Super-Resolution_for_Arbitrary_Blur_Kernels_CVPR_2019_paper.html)|[Code](https://github.com/cszn/DPSR)|
|2019|CVPR|[Dynamic Scene Deblurring With Parameter Selective Sharing and Nested Skip Connections](http://openaccess.thecvf.com/content_CVPR_2019/html/Gao_Dynamic_Scene_Deblurring_With_Parameter_Selective_Sharing_and_Nested_Skip_CVPR_2019_paper.html)||
|2019|CVPR|[Deep Stacked Hierarchical Multi-Patch Network for Image Deblurring](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_Deep_Stacked_Hierarchical_Multi-Patch_Network_for_Image_Deblurring_CVPR_2019_paper.html)|[Code](https://github.com/HongguangZhang/DMPHN-cvpr19-master)|
|2019|CVPR|[Unsupervised Domain-Specific Deblurring via Disentangled Representations](http://openaccess.thecvf.com/content_CVPR_2019/html/Lu_Unsupervised_Domain-Specific_Deblurring_via_Disentangled_Representations_CVPR_2019_paper.html)|[Code](https://github.com/ustclby/Unsupervised-Domain-Specific-Deblurring)|
|2019|CVPR|[Bringing Alive Blurred Moments](http://openaccess.thecvf.com/content_CVPR_2019/html/Purohit_Bringing_Alive_Blurred_Moments_CVPR_2019_paper.html)|[Project page & Results-on-benchmark-datasets](https://github.com/anshulbshah/Blurred-Image-to-Video)|
|2019|CVPR|[Douglas-Rachford Networks: Learning Both the Image Prior and Data Fidelity Terms for Blind Image Deconvolution](http://openaccess.thecvf.com/content_CVPR_2019/html/Aljadaany_Douglas-Rachford_Networks_Learning_Both_the_Image_Prior_and_Data_Fidelity_CVPR_2019_paper.html)||
|2019|ICCV|[DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better](https://arxiv.org/abs/1908.03826)|[Code](https://github.com/TAMU-VITA/DeblurGANv2)|
|2019|ICCV (HIDE)|[Human-Aware Motion Deblurring](https://pdfs.semanticscholar.org/20a4/b3353579525f0b76ec42e17a2284b4453f9a.pdf)||
|2019|BMVC|[Blind image deconvolution using deep generative priors](https://arxiv.org/abs/1802.04073)||
|2019|ACMMM|[Tell Me Where It is Still Blurry: Adversarial Blurred Region Mining and Refining](https://www.iis.sinica.edu.tw/papers/liutyng/22871-F.pdf)||
|2019|IJCV|[Joint Face Hallucination and Deblurring via Structure Generation and Detail Enhancement](https://arxiv.org/abs/1811.09019)|[Code](https://github.com/TAMU-VITA/DeblurGANv2)|
|2020|AAAI|[Learning to Deblur Face Images via Sketch Synthesis](https://aaai.org/ojs/index.php/AAAI/article/view/6818/6672)||
|2020|AAAI|[Region-Adaptive Dense Network for Efficient Motion Deblurring](https://arxiv.org/abs/1903.11394)||
|2020|WACV|[DAVID: Dual-Attentional Video Deblurring](http://openaccess.thecvf.com/content_WACV_2020/html/Wu_DAVID_Dual-Attentional_Video_Deblurring_WACV_2020_paper.html)||
|2020|CVPR|[Neural Blind Deconvolution Using Deep Priors](https://arxiv.org/abs/1908.02197)|[Code](https://github.com/csdwren/SelfDeblur)|
|2020|CVPR|[Spatially-Attentive Patch-Hierarchical Network for Adaptive Motion Deblurring](https://arxiv.org/pdf/2004.05343.pdf)||
|2020|CVPR|[Deblurring by Realistic Blurring](https://arxiv.org/abs/2004.01860)||
|2020|CVPR|[Learning Event-Based Motion Deblurring](https://arxiv.org/abs/2004.05794)||
|2020|CVPR|[Efficient Dynamic Scene Deblurring Using Spatially Variant Deconvolution Network With Optical Flow Guided Training](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Efficient_Dynamic_Scene_Deblurring_Using_Spatially_Variant_Deconvolution_Network_With_CVPR_2020_paper.html)||
|2020|CVPR|[Deblurring using Analysis-Synthesis Networks Pair](https://arxiv.org/abs/2004.02956)||
|2020|ECCV|[Multi-Temporal Recurrent Neural Networks For Progressive Non-Uniform Single Image Deblurring With Incremental Temporal Training](https://arxiv.org/abs/1911.07410)||
|2020|TIP|[Efficient and Interpretable Deep Blind Image Deblurring Via Algorithm Unrolling](https://arxiv.org/pdf/1902.03493.pdf)||
|2020|TIP|[Deblurring Face Images using Uncertainty Guided Multi-Stream Semantic Networks](https://arxiv.org/abs/1907.13106)|[Code](https://github.com/rajeevyasarla/UMSN-Face-Deblurring)|
|2020|TIP|[Dark and bright channel prior embedded network for dynamic scene deblurring](https://www4.comp.polyu.edu.hk/~cslzhang/paper/DBCPeNet_TIP.pdf)|[Code](https://github.com/csjcai/DBCPeNet)|
|2020|TIP|[Dynamic Scene Deblurring by Depth Guided Model](https://faculty.ucmerced.edu/mhyang/papers/tip2020_dynamic_scene_deblurring.pdf)||
|2020|IEEEAccess|[Scale-Iterative Upscaling Network for Image Deblurring](https://ieeexplore.ieee.org/document/8963625)|[Code](https://github.com/minyuanye/SIUN)|
|2020|ACCV|[Human Motion Deblurring using Localized Body Prior](https://openaccess.thecvf.com/content/ACCV2020/html/Lumentut_Human_Motion_Deblurring_using_Localized_Body_Prior_ACCV_2020_paper.html)||
|2020|TCI|[Blind Image Deconvolution using Deep Generative Priors](https://arxiv.org/abs/1802.04073)||
|2020|Arxiv|[Blur Invariant Kernel-Adaptive Network for Single Image Blind deblurring](https://arxiv.org/abs/2007.04543)||
|2021|Arxiv|[Self-supervised Exposure Trajectory Recovery for Dynamic Blur Estimation](https://arxiv.org/abs/2010.02484)||
|2021|Arxiv|[BANet: Blur-aware Attention Networks for Dynamic Scene Deblurring](https://arxiv.org/abs/2101.07518)||
|2021|CVPR|[Multi-Stage Progressive Image Restoration](https://arxiv.org/pdf/2102.02808.pdf)|[Code](https://github.com/swz30/MPRNet)|
|2021|CVPR|[DeFMO: Deblurring and Shape Recovery of Fast Moving Objects](https://arxiv.org/abs/2012.00595)|[Code](https://github.com/rozumden/DeFMO)|
|2021|CVPR|Blind Deblurring for Saturated Images||
|2021|CVPR|Test-Time Fast Adaptation for Dynamic Scene Deblurring via Meta-Auxiliary Learning||
|2021|CVPR|[Explore Image Deblurring via Encoded Blur Kernel Space](https://arxiv.org/abs/2104.00317)||
|2021|JSTS|[Degradation Aware Approach to Image Restoration Using Knowledge Distillation](https://ieeexplore.ieee.org/document/9288928)||
|2021|Arxiv|[Single Image Non-uniform Blur Kernel Estimation via Adaptive Basis Decomposition](https://arxiv.org/abs/2102.01026)||
|2021|Arxiv|[Clean Images are Hard to Reblur: A New Clue for Deblurring](https://arxiv.org/pdf/2104.12665.pdf)||
|2021|CVIU|[Single-image deblurring with neural networks: A comparative survey](https://www.sciencedirect.com/science/article/abs/pii/S1077314220301533?dgcid=rss_sd_all)||
|2021|arxiv|[Blind Motion Deblurring Super-Resolution: When Dynamic Spatio-Temporal Learning Meets Static Image Understanding](https://arxiv.org/pdf/2105.13077.pdf)||

## Non-Blind-Deblurring
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2006|IJCV|[Image deblurring in the presence of impulsive noise](https://link.springer.com/article/10.1007/s11263-006-6468-1)||
|2009|NIPS|[Fast image deconvolution using hyper-laplacian priors](http://cs.nyu.edu/~dilip/research/papers/fid_nips09.pdf)|[Code & Project page](https://dilipkay.wordpress.com/fast-deconvolution/)|
|2011|PAMI|[Richardson-Lucy Deblurring for Scenes under a Projective Motion Path](https://ieeexplore.ieee.org/document/5674049)||
|2011|ICCV|[Handling outliers in non-blind image deconvolution](http://cg.postech.ac.kr/papers/deconv_outliers.pdf)|[Code](https://github.com/CoupeLibrary/handleoutlier)|
|2011|ICCV|[From learning models of natural image patches to whole image restoration](http://people.ee.duke.edu/~lcarin/EPLICCVCameraReady.pdf)|[Code](http://people.csail.mit.edu/danielzoran/)|
|2012|TIP|[Bm3d frames and variational image deblurring](https://www.cs.tut.fi/~foi/GCF-BM3D/BM3DframesDeblur-Danielyan.pdf)||
|2012|TIP|[Robust image deblurring with an inaccurate blur kernel](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.716.1055&rep=rep1&type=pdf) [Code](https://blog.nus.edu.sg/matjh/download/)|
|2013|CVPR|[A machine learning approach for non-blind image deconvolution](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Schuler_A_Machine_Learning_2013_CVPR_paper.pdf)|[Code & Project page](http://webdav.is.mpg.de/pixel/neural_deconvolution/)|
|2013|CVPR|[Discriminative non-blind deblurring](http://openaccess.thecvf.com/content_cvpr_2013/papers/Schmidt_Discriminative_Non-blind_Deblurring_2013_CVPR_paper.pdf)|[Code](https://www.visinf.tu-darmstadt.de/vi_research/code/index.en.jsp#discriminative_deblurring)|
|2014|TIP|[A general framework for regularized, similarity-based image restoration](http://www.academia.edu/download/42621942/A_General_Framework_for_Regularized_Simi20160212-19526-i3txol.pdf) [Code & Project page](http://alumni.soe.ucsc.edu/~aminkh/KernelRestoration.html)|
|2014|NIPS|[Deep convolutional neural network for image deconvolution](http://www.cse.cuhk.edu.hk/leojia/papers/deconv_nips14.pdf)|[Code & Project page](http://lxu.me/projects/dcnn/)|
|2014|CVPR|[Shrinkage fields for effective image restoration](http://research.uweschmidt.org/pubs/cvpr14schmidt.pdf)|[Code](https://github.com/uschmidt83/shrinkage-fields)|
|2014|ECCV|[Good Image Priors for Non-blind Deconvolution: Generic vs Specific](http://cs.brown.edu/~lbsun/GoodPriors2014/goodpriors_eccv2014.pdf)|[Project page](http://cs.brown.edu/~lbsun/GoodPriors2014/goodpriors2014eccv.html)|
|2016|CVIP|[Fast Non-Blind Image De-blurring With Sparse Priors](https://link.springer.com/chapter/10.1007/978-981-10-2104-6_56)||
|2017|TIP|[Partial Deconvolution With Inaccurate Blur Kernel](https://ieeexplore.ieee.org/document/8071032)||
|2017|ICCP|[Fast non-blind deconvolution via regularized residual networks with long/short skip-connections](http://cg.postech.ac.kr/papers/skipConnect.pdf)|[Code](https://github.com/HyeongseokSon1/CNN_deconvolution), [Project Page](http://cg.postech.ac.kr/research/resnet_deconvolution/)|
|2017|CVPR|[Noise-Blind Image Deblurring](http://openaccess.thecvf.com/content_cvpr_2017/html/Jin_Noise-Blind_Image_Deblurring_CVPR_2017_paper.html)||
|2017|CVPR|[Learning Deep CNN Denoiser Prior for Image Restoration](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Deep_CNN_CVPR_2017_paper.html)|[Code](https://github.com/cszn/ircnn)|
|2017|CVPR|[Learning Fully Convolutional Networks for Iterative Non-blind Deconvolution](https://arxiv.org/pdf/1611.06495)|[Code](https://github.com/zhjwustc/cvpr17_iter_deblur_testing_matconvnet)|
|2017|ICCV|[Learning proximal operators: Using denoising networks for regularizing inverse imaging problems](https://arxiv.org/abs/1704.03488)||
|2017|ICCV|[Learning to push the limits of efficient fft-based image deconvolution](http://research.uweschmidt.org/pubs/iccv17kruse.pdf)|[Code](https://github.com/uschmidt83/fourier-deconvolution-network)|
|2017|NIPS|[Deep Mean-Shift Priors for Image Restoration](https://papers.nips.cc/paper/6678-deep-mean-shift-priors-for-image-restoration.pdf)|[Code](https://github.com/siavashbigdeli/DMSP)|
|2018|ICIP|[Modeling Realistic Degradations in Non-Blind Deconvolution](https://arxiv.org/abs/1806.01097)||
|2018|CVPR|[Non-blind Deblurring: Handling Kernel Uncertainty with CNNs](http://openaccess.thecvf.com/content_cvpr_2018/html/Vasu_Non-Blind_Deblurring_Handling_CVPR_2018_paper.html)|[Project page & Results-on-benchmark-datasets](https://github.com/subeeshvasu/2018_subeesh_nbd_cvpr)|
|2018|CVPR|[Deep image prior](https://arxiv.org/abs/1711.10925)|[Code](https://github.com/DmitryUlyanov/deep-image-prior)|
|2018|ECCV|[Learning Data Terms for Non-blind Deblurring](http://openaccess.thecvf.com/content_ECCV_2018/html/Jiangxin_Dong_Learning_Data_Terms_ECCV_2018_paper.html)||
|2018|NIPS|[Deep Non-Blind Deconvolution via Generalized Low-Rank Approximation](https://papers.nips.cc/paper/7313-deep-non-blind-deconvolution-via-generalized-low-rank-approximation.pdf)|[Code](https://github.com/rwenqi/NBD-GLRA)|
|2019|ICLR|[Deep decoder: Concise image representations from untrained non-convolutional networks](https://arxiv.org/abs/1810.03982)|[Code](https://github.com/reinhardh/supplement_deep_decoder)|
|2019|ICCVW|[Image deconvolution with deep image and kernel priors](https://arxiv.org/abs/1910.08386)||
|2019|TPAMI|[Denoising prior driven deep neural network for image restoration](https://arxiv.org/abs/1801.06756)||
|2020|CVPR|[Variational-EM-Based Deep Learning for Noise-Blind Image Deblurring](https://github.com/ysnan/VEM-NBD/blob/master/paper/vem_deconv.pdf)|[Project page & Results-on-benchmark-datasets](https://github.com/ysnan/VEM-NBD)|
|2020|CVPR|[Deep Learning for Handling Kernel/model Uncertainty in Image Deconvolution](https://github.com/ysnan/NBD_KerUnc/blob/master/paper/kn.pdf)|[Project page & Results-on-benchmark-datasets](https://github.com/ysnan/NBD_KerUnc)|
|2020|ECCV|[End-to-end interpretable learning of non-blind image deblurring](https://arxiv.org/abs/2007.01769)||
|2020|EUSIPCO|[Bp-dip: A backprojection based deep image prior](https://arxiv.org/abs/2003.05417)|[Code](https://github.com/jennyzu/BP-DIP-deblurring)|
|2020|NIPS|[Deep Wiener Deconvolution: Wiener Meets Deep Learning for Image Deblurring](https://papers.nips.cc/paper/2020/file/0b8aff0438617c055eb55f0ba5d226fa-Paper.pdf)|[Code](https://gitlab.mpi-klsb.mpg.de/jdong/dwdn)|
|2020|TNLS|[Learning deep gradient descent optimization for image deconvolution](https://arxiv.org/abs/1804.03368)|[Code](https://github.com/donggong1/learn-optimizer-rgdn)|
|2020|TCI|[Neumann networks for linear inverse problems in imaging](https://arxiv.org/abs/1901.03707)|[Code](https://github.com/dgilton/neumann_networks_code)|
|2020|Arxiv|[The Maximum Entropy on the Mean Method for Image Deblurring](https://arxiv.org/pdf/2002.10434.pdf)||
|2021|CVPR|Learning Spatially-Variant MAP Models for Non-Blind Image Deblurring||
|2021|CVPR|Learning a Non-Blind Deblurring Network for Night Blurry Images||
|2021|TNNLS|[Nonblind Image Deblurring via Deep Learning in Complex Field](https://ieeexplore.ieee.org/abstract/document/9404870)||

## (Multi-image/Video)-Motion-Deblurring
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2007|TOG|[Image Deblurring with Blurred/Noisy Image Pairs](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/Deblurring_SIGGRAPH07.pdf)||
|2008|CVPR|[Robust dual motion deblurring](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.443.6370&rep=rep1&type=pdf)||
|2009|JCP|[Blind motion deblurring using multiple images](https://www.sciencedirect.com/science/article/pii/S0021999109001867)||
|2010|CVPR|[Robust flash deblurring](https://ieeexplore.ieee.org/document/5539941)||
|2010|CVPR|[Efficient filter flow for space-variant multiframe blind deconvolution](http://suvrit.de/papers/cvpr10.pdf)||
|2012|ECCV|[Deconvolving PSFs for A Better Motion Deblurring using Multiple Images](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.259.6526&rep=rep1&type=pdf)||
|2012|TIP|[Robust multichannel blind deconvolution via fast alternating minimization](https://users.soe.ucsc.edu/~milanfar/publications/journal/MCBD.pdf)||
|2012|CGF|[Registration Based Non-uniform Motion Deblurring](http://cg.postech.ac.kr/papers/registration.pdf)||
|2012|TOG|[Video deblurring for hand-held cameras using patch-based synthesis](https://www.juew.org/publication/video_deblur.pdf)|[Project page](http://cg.postech.ac.kr/research/video_deblur/)|
|2013|CVPR|[Multi-image Blind Deblurring Using a Coupled Adaptive Sparse Prior](http://openaccess.thecvf.com/content_cvpr_2013/html/Zhang_Multi-image_Blind_Deblurring_2013_CVPR_paper.html)|[Code & Project page](https://sites.google.com/site/hczhang1/projects/sparse-blind-deblurring)|
|2014|CVPR|[Multi-Shot Imaging: Joint Alignment, Deblurring and Resolution Enhancement](http://openaccess.thecvf.com/content_cvpr_2014/html/Zhang_Multi-Shot_Imaging_Joint_2014_CVPR_paper.html)|[Project page](https://sites.google.com/site/hczhang1/projects/multi-shot-imaging)|
|2014|CVPR|[Gyro-Based Multi-Image Deconvolution for Removing Handshake Blur](http://graphics.stanford.edu/papers/gyrodeblur/gyrodeblur_park_cvpr14.pdf)|[Project Page](http://graphics.stanford.edu/papers/gyrodeblur/)|
|2014|ECCV|[Modeling Blurred Video with Layers](http://files.is.tue.mpg.de/black/papers/WulffECCV2014.pdf)|[Project page, Results & Dataset](http://ps.is.tuebingen.mpg.de/research_projects/motion-blur-in-layers)|
|2015|CVPR|[Burst Deblurring: Removing Camera Shake Through Fourier Burst Accumulation](http://dev.ipol.im/~mdelbra/fba/FBA_cvpr2015_preprint.pdf)||
|2015|TCI|[Hand-held video deblurring via efficient fourier aggregation](http://arxiv.org/pdf/1509.05251)|[Project page & Results](http://iie.fing.edu.uy/~mdelbra/videoFA/)||
|2015|TIP|[Removing camera shake via weighted fourier burst accumulation](https://arxiv.org/abs/1505.02731)||
|2015|CVPR|[Generalized Video Deblurring for Dynamic Scenes](http://cv.snu.ac.kr/publication/conf/2015/VD_CVPR2015.pdf)|[Code & Project page](https://cv.snu.ac.kr/research/~VD/)||
|2015|CVPR|[Intra-Frame Deblurring by Leveraging Inter-Frame Camera Motion](http://openaccess.thecvf.com/content_cvpr_2015/html/Zhang_Intra-Frame_Deblurring_by_2015_CVPR_paper.html)|[Project page](https://sites.google.com/site/hczhang1/projects/video_deblur)|
|2016|ECCV|[Stereo video deblurring](https://arxiv.org/abs/1607.08421)||
|2017|CVPR|[Simultaneous stereo video deblurring and scene flow estimation](https://arxiv.org/abs/1704.03273)||
|2017|CVPR|[Deep Video Deblurring for Hand-Held Cameras](http://openaccess.thecvf.com/content_cvpr_2017/html/Su_Deep_Video_Deblurring_CVPR_2017_paper.html)|[Code](https://github.com/shuochsu/DeepVideoDeblurring)|[Project page](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/)|
|2017|CVPR|[Light Field Blind Motion Deblurring](http://openaccess.thecvf.com/content_cvpr_2017/html/Srinivasan_Light_Field_Blind_CVPR_2017_paper.html)|[code](https://github.com/pratulsrinivasan/Light_Field_Blind_Motion_Deblurring)|
|2017|ICCV|[Video Deblurring via Semantic Segmentation and Pixel-Wise Non-Linear Kernel](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ren_Video_Deblurring_via_ICCV_2017_paper.pdf)|[Project page](https://sites.google.com/site/renwenqi888/research/deblurring/pwnlk)|
|2017|ICCV|[Online Video Deblurring via Dynamic Temporal Blending Network](http://openaccess.thecvf.com/content_ICCV_2017/papers/Kim_Online_Video_Deblurring_ICCV_2017_paper.pdf)|[Code](https://sites.google.com/site/lliger9/publications)|
|2018|ECCV|[Burst Image Deblurring Using Permutation Invariant Convolutional Neural Networks](http://openaccess.thecvf.com/content_ECCV_2018/html/Miika_Aittala_Burst_Image_Deblurring_ECCV_2018_paper.html)|[Project page](http://people.csail.mit.edu/miika/eccv18_deblur/)|
|2018|ECCV|[Joint Blind Motion Deblurring and Depth Estimation of Light Field](http://openaccess.thecvf.com/content_ECCV_2018/html/Dongwoo_Lee_Joint_Blind_Motion_ECCV_2018_paper.html)||
|2018|ICCP|[Reblur2deblur: Deblurring videos via self-supervised learning](https://arxiv.org/pdf/1801.05117.pdf)||
|2018|Arxiv|[LSD-Joint Denoising and Deblurring of Short and Long Exposure Images with Convolutional Neural Networks](https://arxiv.org/abs/1811.09485)||
|2019|TIP|[Adversarial Spatio-Temporal Learning for Video Deblurring](https://arxiv.org/abs/1804.00533)|[Code](https://github.com/themathgeek13/STdeblur)|[Project page](https://github.com/JLtwoP/Adversarial-Spatio-Temporal-Learning-for-Video-Deblurring)|
|2019|CVPR|[Recurrent Neural Networks With Intra-Frame Iterations for Video Deblurring](http://openaccess.thecvf.com/content_CVPR_2019/html/Nah_Recurrent_Neural_Networks_With_Intra-Frame_Iterations_for_Video_Deblurring_CVPR_2019_paper.html)||
|2019|CVPR|[DAVANet: Stereo Deblurring With View Aggregation](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_DAVANet_Stereo_Deblurring_With_View_Aggregation_CVPR_2019_paper.html)|[Code](https://github.com/sczhou/DAVANet)|
|2019|CVPR_W|[A Deep Motion Deblurring Network based on Per-Pixel Adaptive Kernels with Residual Down-Up and Up-Down Modules](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Sim_A_Deep_Motion_Deblurring_Network_Based_on_Per-Pixel_Adaptive_Kernels_CVPRW_2019_paper.html)||
|2019|ICCV|[Spatio-Temporal Filter Adaptive Network for Video Deblurring](https://arxiv.org/abs/1904.12257)|[Project page](https://shangchenzhou.com/projects/stfan/), [Code](https://github.com/sczhou/STFAN)|
|2019|ICCV|[Face Video Deblurring using 3D Facial Priors](https://openaccess.thecvf.com/content_ICCV_2019/html/Ren_Face_Video_Deblurring_Using_3D_Facial_Priors_ICCV_2019_paper.html)|[Code](https://github.com/rwenqi/3Dfacedeblurring)|
|2019|SPL|[Deep Recurrent Network for Fast and Full-Resolution Light Field Deblurring](https://arxiv.org/abs/1904.00352)||
|2019|ICCV_W|[Deep Video Deblurring: The Devil is in the Details](https://arxiv.org/abs/1909.12196)|[Code](https://github.com/visinf/deblur-devil)|
|2020|CVPR|[Cascaded Deep Video Deblurring Using Temporal Sharpness Prior](https://arxiv.org/abs/2004.02501)|[Code](https://github.com/csbhr/CDVD-TSP)|[Project Page](https://baihaoran.xyz/projects/cdvd-tsp/index.html)|
|2020|CVPR|[Blurry Video Frame Interpolation](https://arxiv.org/abs/2002.12259)|[Code](https://github.com/laomao0/BIN)|
|2020|ECCV|[Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5116_ECCV_2020_paper.php)|[Code](https://github.com/zzh-tech/ESTRNN)|
|2020|ECCV|[Learning Event-Driven Video Deblurring and Interpolation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/671_ECCV_2020_paper.php)||
|2020|TIP|[Blur Removal Via Blurred-Noisy Image Pair](https://arxiv.org/abs/1903.10667)||
|2020|TCSVT|[Recursive Neural Network for Video Deblurring](https://ieeexplore.ieee.org/abstract/document/9247314)||
|2021|AAAI|[Motion-blurred Video Interpolation and Extrapolation](https://arxiv.org/pdf/2103.02984.pdf)||
|2021|CVPR|[Gated Spatio-Temporal Attention-Guided Video Deblurring]()||
|2021|CVPR|[ARVo: Learning All-Range Volumetric Correspondence for Video Deblurring](https://arxiv.org/abs/2103.04260)||



## Challenges on Motion Deblurring

|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2019|CVPR_W|[NTIRE 2019 Challenge on Video Deblurring: Methods and Results](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Deblurring_Methods_and_Results_CVPRW_2019_paper.html)||
|2019|CVPR_W|[NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Deblurring_and_Super-Resolution_Dataset_and_CVPRW_2019_paper.html)||
|2019|CVPR_W|[EDVR: Video Restoration with Enhanced Deformable Convolutional Networks](https://arxiv.org/abs/1905.02716)|[Code-Pytorch](https://github.com/xinntao/EDVR)|[Project page](https://xinntao.github.io/projects/EDVR)|
|2020|CVPR_W|[Ntire 2020 challenge on image and video deblurring](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Nah_NTIRE_2020_Challenge_on_Image_and_Video_Deblurring_CVPRW_2020_paper.pdf)||
|2020|CVPR_W|[Deploying Image Deblurring across Mobile Devices: A Perspective of Quality and Latency](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Chiang_Deploying_Image_Deblurring_Across_Mobile_Devices_A_Perspective_of_Quality_CVPRW_2020_paper.pdf)||
|2020|CVPR_W|[High-Resolution Dual-Stage Multi-Level Feature Aggregation for Single Image and Video Deblurring](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Brehm_High-Resolution_Dual-Stage_Multi-Level_Feature_Aggregation_for_Single_Image_and_Video_CVPRW_2020_paper.pdf)||


## Other Closely Related Works
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2013|TOG|[A No-Reference Metric for Evaluating The Quality of Motion Deblurring](https://gfx.cs.princeton.edu/pubs/Liu_2013_ANM/sa13.pdf)|[Code & Project Page](https://gfx.cs.princeton.edu/pubs/Liu_2013_ANM/index.php)|
|2018|CVPR|[Learning to extract a video sequence from a single motion-blurred image](http://openaccess.thecvf.com/content_cvpr_2018/html/Jin_Learning_to_Extract_CVPR_2018_paper.html)||[Code](https://github.com/MeiguangJin/Learning-to-Extract-a-Video-Sequence-from-a-Single-Motion-Blurred-Image)|
|2019|CVPR|[Bringing a Blurry Frame Alive at High Frame-Rate With an Event Camera](http://openaccess.thecvf.com/content_CVPR_2019/html/Pan_Bringing_a_Blurry_Frame_Alive_at_High_Frame-Rate_With_an_CVPR_2019_paper.html)|[Code](https://github.com/panpanfei/Bringing-a-Blurry-Frame-Alive-at-High-Frame-Rate-with-an-Event-Camera)|
|2019|CVPR|[Learning to Extract Flawless Slow Motion From Blurry Videos](http://openaccess.thecvf.com/content_CVPR_2019/html/Jin_Learning_to_Extract_Flawless_Slow_Motion_From_Blurry_Videos_CVPR_2019_paper.html)|[Code](https://github.com/MeiguangJin/slow-motion)|
|2019|CVPR|[Learning to Synthesize Motion Blur](http://openaccess.thecvf.com/content_CVPR_2019/html/Brooks_Learning_to_Synthesize_Motion_Blur_CVPR_2019_paper.html)|[Code](https://github.com/google-research/google-research/tree/master/motion_blur), [Project page](http://timothybrooks.com/tech/motion-blur/)|
|2019|CVPR|[World from blur](https://openaccess.thecvf.com/content_CVPR_2019/papers/Qiu_World_From_Blur_CVPR_2019_paper.pdf)||
|2019|ICCV|[FAB: A Robust Facial Landmark Detection Framework for Motion-Blurred Videos](https://arxiv.org/abs/1910.12100)|[Code](https://github.com/KeqiangSun/FAB)|
|2019|ICCV|[Visual Deprojection: Probabilistic Recovery of Collapsed Dimensions](https://arxiv.org/abs/1909.00475)||
|2020|CVPR-W|[Photosequencing of Motion Blur using Short and Long Exposures](https://arxiv.org/abs/1912.06102)|[Project Page](https://apvijay.github.io/photoseq_blur.html)|
|2020|ACM-MM|Every Moment Matters: Detail-Aware Networks to Bring a Blurry Image Alive||
|2020|NIPS|[Watch out! Motion is Blurring Blurring the Vision of Your Deep Neural Networks](https://proceedings.neurips.cc/paper/2020/file/0a73de68f10e15626eb98701ecf03adb-Paper.pdf)|[Code](https://github.com/tsingqguo/ABBA)|
|2020|Arxiv|[Improved Handling of Motion Blur in Online Object Detection](https://arxiv.org/abs/2011.14448)||
|2021|Arxiv|[Geometric Moment Invariants to Motion Blur](https://arxiv.org/abs/2101.08647v2)||
|2021|AAAI|[Optical Flow Estimation from a Single Motion-blurred Image](https://arxiv.org/pdf/2103.02996.pdf)||
|2021|CVPR|[Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes](https://arxiv.org/abs/2104.01601)|[Code](https://github.com/zzh-tech/RSCD)||
|2021|CVPR|[Improved Handling of Motion Blur in Online Object Detection](https://arxiv.org/abs/2011.14448)||
|2021|CVPR|[Blur, Noise, and Compression Robust Generative Adversarial Networks](https://arxiv.org/abs/2003.07849)||

## Defocus Deblurring and Potential Datasets
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2009|ICCP|[What are Good Apertures for Defocus Deblurring?](https://ieeexplore.ieee.org/document/5559018)||
|2012|ICASSP|[Spatially-varying out-of-focus image deblurring with L1-2 optimization and a guided blur map](https://ieeexplore.ieee.org/document/6288071)||
|2013|ICASSP|[Removing out-of-focus blur from similar image pairs](https://ieeexplore.ieee.org/document/6637925)||
|2014|CVPR|[Discriminative Blur Detection Features](http://www.shijianping.me/blur_cvpr14.pdf)|[Project Page](http://www.cse.cuhk.edu.hk/~leojia/projects/dblurdetect/index.html)|
|2015|CVPR|[Just Noticeable Defocus Blur Detection and Estimation](http://shijianping.me/jnb/papers/jnbdetection_final.pdf)|[Project Page](http://shijianping.me/jnb/index.html)|
|2016||[Spatially Variant Defocus Blur Map Estimation and Deblurring from a Single Image](https://www.sciencedirect.com/science/article/pii/S1047320316000031)|[Code](https://github.com/ZHANGXinxinPKU/defocus-deblurring)|
|2017|BMVC|[Depth Estimation and Blur Removal from a Single Out-of-focus Image](https://saeed-anwar.github.io/papers/BMVC17-depth.pdf)||
|2017|CVPR|[Spatially-Varying Blur Detection Based on Multiscale Fused and Sorted Transform Coefficients of Gradient Magnitudes](http://openaccess.thecvf.com/content_cvpr_2017/html/Golestaneh_Spatially-Varying_Blur_Detection_CVPR_2017_paper.html)|[Code](https://github.com/isalirezag/HiFST)|
|2017|ICCV|[Learning to Synthesize a 4D RGBD Light Field from a Single Image](http://openaccess.thecvf.com/content_iccv_2017/html/Srinivasan_Learning_to_Synthesize_ICCV_2017_paper.html)|[Dataset and Project Page](https://github.com/pratulsrinivasan/Local_Light_Field_Synthesis)|
|2018|ECCV|[Refocusgan: Scene refocusing using a single image](https://openaccess.thecvf.com/content_ECCV_2018/papers/Parikshit_Sakurikar_Single_Image_Scene_ECCV_2018_paper.pdf)||
|2018|ECCV_W|[Deep Depth from Defocus: how can defocus blur improve 3D estimation using dense neural networks?](http://openaccess.thecvf.com/content_eccv_2018_workshops/w3/html/Carvalho_Deep_Depth_from_Defocus_how_can_defocus_blur_improve_3D_ECCVW_2018_paper.html)|[Code & Dataset](https://github.com/marcelampc/d3net_depth_estimation)|
|2018|PG|[Defocus and Motion Blur Detection with Deep Contextual Features](http://cg.postech.ac.kr/papers/Kim2018Defocus.pdf)|[Code & Dataset](https://github.com/HyeongseokSon1/deep_blur_detection_and_classification)|
|2018|TIP|[Edge-based defocus blur estimation with adaptive scale selection](https://ieeexplore.ieee.org/document/8101511)|[Code](https://github.com/alikaraali/TIP2018-Edge-Based-Defocus-Blur-Estimation-With-Adaptive-Scale-Selection)|
|2019|CVPR|[Deep Defocus Map Estimation using Domain Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/html/Lee_Deep_Defocus_Map_Estimation_Using_Domain_Adaptation_CVPR_2019_paper.html)|[Code & Dataset](https://github.com/codeslake/DMENet)|
|2019|CVPR|[DeFusionNET: Defocus Blur Detection via Recurrently Fusing and Refining Multi-Scale Deep Features](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_DeFusionNET_Defocus_Blur_Detection_via_Recurrently_Fusing_and_Refining_Multi-Scale_CVPR_2019_paper.pdf)||
|2020|ECCV|[Defocus Deblurring Using Dual-Pixel Data](https://arxiv.org/abs/2005.00305)||
|2020|ECCV|[Rethinking the Defocus Blur Detection Problem and A Real-Time Deep DBD Model](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1182_ECCV_2020_paper.php)||
|2020|ECCV|[Defocus Blur Detection via Depth Distillation](https://arxiv.org/abs/2007.08113)|[Code](https://github.com/vinthony/depth-distillation)|
|2020|Arxiv|[CycleGAN with a Blur Kernel for Deconvolution Microscopy: Optimal Transport Geometry](https://arxiv.org/abs/1908.09414)||
|2020|Arxiv|[Deep Multi-Scale Feature Learning for Defocus Blur Estimation](https://arxiv.org/abs/2009.11939)||
|2021|Arxiv|[Defocus Blur Detection via Salient Region Detection Prior](https://arxiv.org/abs/2011.09677)||
|2021|Arxiv|[Learning to Estimate Kernel Scale and Orientation of Defocus Blur with Asymmetric Coded Aperture](https://arxiv.org/abs/2103.05843)||
|2021|CVPR|Iterative Filter Adaptive Network for Single Image Defocus Deblurring||
|2021|CVPR|Self-Generated Defocus Blur Detection via Dual Adversarial Discriminators||

## Benchmark Datasets on Motion Deblurring
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2009|CVPR|[Understanding and evaluating blind deconvolution algorithms](http://webee.technion.ac.il/people/anat.levin/papers/deconvLevinEtalCVPR09.pdf)|[Dataset](http://webee.technion.ac.il/people/anat.levin/papers/LevinEtalCVPR09Data.rar)|
|2012|ECCV|[Recording and playback of camera shake: benchmarking blind deconvolution with a real-world database](http://webdav.is.mpg.de/pixel/benchmark4camerashake/src_files/Pdf/Koehler_ECCV2012_Benchmark.pdf)|[Dataset](http://webdav.is.mpg.de/pixel/benchmark4camerashake/)|
|2013|ICCP|[Edge-based blur kernel estimation using patch priors](http://cs.brown.edu/~lbsun/deblur2013/patchdeblur_iccp2013.pdf)|[Dataset](http://cs.brown.edu/~lbsun/deblur2013/deblur2013iccp.html)|
|2016|CVPR|[A Comparative Study for Single Image Blind Deblurring](http://vllab.ucmerced.edu/wlai24/cvpr16_deblur_study/paper/cvpr16_deblur_study.pdf)|[Dataset](http://vllab.ucmerced.edu/wlai24/cvpr16_deblur_study/)|
|2017|CVPR (GOPRO)|[Deep multi-scale convolutional neural network for dynamic scene deblurring](http://zpascal.net/cvpr2017/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)|[Dataset](https://github.com/SeungjunNah/DeepDeblur_release)|
|2017|CVPR|[Deep Video Deblurring for Hand-Held Cameras](http://openaccess.thecvf.com/content_cvpr_2017/html/Su_Deep_Video_Deblurring_CVPR_2017_paper.html)|[Dataset](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/)|
|2019|CVPR (Stereo Blur Dataset)|[Stereo Deblurring With View Aggregation](http://openaccess.thecvf.com/content_CVPR_2019/html/Zhou_DAVANet_Stereo_Deblurring_With_View_Aggregation_CVPR_2019_paper.html)|[Dataset](https://stereoblur.shangchenzhou.com/)|
|2019|CVPR_W (REDS)|[NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study](http://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Nah_NTIRE_2019_Challenge_on_Video_Deblurring_and_Super-Resolution_Dataset_and_CVPRW_2019_paper.html)|[Dataset](https://seungjunnah.github.io/Datasets/reds)|
|2019|ICCV (HIDE)|[Human-Aware Motion Deblurring](https://pdfs.semanticscholar.org/20a4/b3353579525f0b76ec42e17a2284b4453f9a.pdf)|[Dataset](https://github.com/joanshen0508/HA_deblur)|
|2020|CVPR|[Deblurring by Realistic Blurring](https://arxiv.org/abs/2004.01860)|[Dataset](https://sites.google.com/view/kaihaozhang/publications)|
|2020|CVPR|[Learning Event-Based Motion Deblurring](https://arxiv.org/abs/2004.05794)||
|2020|ECCV (BSD)|[Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf)|[Dataset](https://github.com/zzh-tech/ESTRNN)|
|2020|ECCV|[Real-World Blur Dataset for Learning and Benchmarking Deblurring Algorithms](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700188.pdf)|[Code & Dataset](http://cg.postech.ac.kr/research/realblur/)|
|2021|CVPR|[Towards Rolling Shutter Correction and Deblurring in Dynamic Scenes](https://arxiv.org/abs/2104.01601)|[Dataset](https://github.com/zzh-tech/RSCD)||


Abbreviations:

+ DL -> Deep Learning
+ non-DL -> non-Deep Learning
