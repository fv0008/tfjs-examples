/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as https from 'https';
import * as tf from '@tensorflow/tfjs';

// TODO(cais): Support user-supplied text data.
export const TEXT_DATA_URLS = {
  'names': {
    url:
        //'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/nietzsche.txt',
        './names.txt',
    //needle: 'Nietzsche'
    needle: 'names'
    
  },
  'poems': {
    url:
        //'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/t1.verne.txt',
        './poems.txt',
    //needle: 'Jules Verne'
    needle: 'poems'
  },
  /*
  'shakespeare': {
    url:
        'https://storage.googleapis.com/tfjs-examples/lstm-text-generation/data/t8.shakespeare.txt',
    needle: 'Shakespeare'
  },
  'tfjs-code': {
    url: 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.11.7/dist/tf.js',
    needle: 'TensorFlow.js Code (Compiled, 0.11.7)'
  }
  */
}

/**
 * A class for text data.
 *
 * This class manages the following:
 *
 * - Converting training data (as a string) into one-hot encoded vectors.
 * - Drawing random slices from the training data. This is useful for training
 *   models and obtaining the seed text for model-based text generation.
 */
export class TextData {
  /**
   * Constructor of TextData.
   *
   * @param {string} dataIdentifier An identifier for this instance of TextData.
   * @param {string} textString The training text data.
   * @param {number} sampleLen Length of each training example, i.e., the input
   *   sequence length expected by the LSTM model.
   * @param {number} sampleStep How many characters to skip when going from one
   *   example of the training data (in `textString`) to the next.
   */
  constructor(dataIdentifier, textString, sampleLen, sampleStep) {
    tf.util.assert(
        sampleLen > 0,
        `Expected sampleLen to be a positive integer, but got ${sampleLen}`);
    tf.util.assert(
        sampleStep > 0,
        `Expected sampleStep to be a positive integer, but got ${sampleStep}`);

    if (!dataIdentifier) {
      throw new Error('Model identifier is not provided.');
    }

    this.dataIdentifier_ = dataIdentifier;

    this.textString_ = textString;
    this.textLen_ = textString.length;
    this.sampleLen_ = sampleLen;
    this.sampleStep_ = sampleStep;
    this.textlist = []

    this.splitAllText()
    this.getCharSet_();
    this.convertAllTextToIndices_();
    
  }

  splitAllText(){
    var test = this.textString_
    test.replace(" ","")
    test.replace("_","")
    this.textlist = test.split(/[\n\t:]/)
  }


  /**
   * Get data identifier.
   *
   * @returns {string} The data identifier.
   */
  dataIdentifier() {
    return this.dataIdentifier_;
  }

  /**
   * Get length of the training text data.
   *
   * @returns {number} Length of training text data.
   */
  textLen() {
    return this.textLen_;
  }

  /**
   * Get the length of each training example.
   */
  sampleLen() {
    return this.sampleLen_;
  }

  /**
   * Get the size of the character set.
   *
   * @returns {number} Size of the character set, i.e., how many unique
   *   characters there are in the training text data.
   */
  charSetSize() {
    return this.charSetSize_;
  }

  /**
   * Generate the next epoch of data for training models.
   *
   * @param {number} numExamples Number examples to generate.
   * @returns {[tf.Tensor, tf.Tensor]} `xs` and `ys` Tensors.
   *   `xs` has the shape of `[numExamples, this.sampleLen, this.charSetSize]`.
   *   `ys` has the shape of `[numExamples, this.charSetSize]`.
   */
  nextDataEpoch(index,numExamples) {
    this.generateExampleBeginIndices_();

    if (numExamples == null) {
      numExamples = this.exampleBeginIndices_.length;
    }
    var textExamples = ""
    for(let le = 0; le < this.textlist.length;le++)
    {
      textExamples = textExamples + this.textlist[le]
    }

    const xsBuffer = new tf.TensorBuffer([
      numExamples, this.sampleLen_,  this.charSetSize_]);
    const ysBuffer  = new tf.TensorBuffer([numExamples,  this.charSetSize_]);

    var beginIndex = 0
    for(let le = 0; le < index * numExamples;le++)
    {
        beginIndex = beginIndex + this.textlist[le].length
    }

    for (let i = 0; i < numExamples; ++i) {
      var curText = this.textlist[index * numExamples + i]
      
      for (let j = 0; j < curText.length; ++j) {
        xsBuffer.set(1, i, j,  this.indices_[ beginIndex+j]);
        //console.log("j:"+j,this.charSet_[this.indices_[ beginIndex+j]])
      }
      beginIndex = beginIndex + curText.length
      ysBuffer.set(1, i,  this.indices_[beginIndex]);
      //console.log("i:"+i,this.charSet_[this.indices_[beginIndex]])
    }
    /*
    const xsBuffer = new tf.TensorBuffer([
        numExamples, this.sampleLen_, this.charSetSize_]);
    const ysBuffer  = new tf.TensorBuffer([numExamples, this.charSetSize_]);
    for (let i = 0; i < numExamples; ++i) {
      const beginIndex = this.exampleBeginIndices_[
          this.examplePosition_ % this.exampleBeginIndices_.length];
      for (let j = 0; j < this.sampleLen_; ++j) {
        xsBuffer.set(1, i, j, this.indices_[beginIndex + j]);
      }
      ysBuffer.set(1, i, this.indices_[beginIndex + this.sampleLen_]);
      this.examplePosition_++;
    }
    */
    return [xsBuffer.toTensor(), ysBuffer.toTensor()];
  }

  /**
   * Get the unique character at given index from the character set.
   *
   * @param {number} index
   * @returns {string} The unique character at `index` of the character set.
   */
  getFromCharSet(index) {
    return this.charSet_[index];
  }

  /**
   * Convert text string to integer indices.
   *
   * @param {string} text Input text.
   * @returns {number[]} Indices of the characters of `text`.
   */
  textToIndices(text) {
    const indices = [];
    for (let i = 0; i < text.length; ++i) {
      indices.push(this.charSet_.indexOf(text[i]));
    }
    return indices;
  }

  /**
   * Get a random slice of text data.
   *
   * @returns {[string, number[]} The string and index representation of the
   *   same slice.
   */
  getRandomSlice() {
    /*
    const startIndex =
        Math.round(Math.random() * (this.textLen_ - this.sampleLen_ - 1));
    const textSlice = this.slice_(startIndex, startIndex + this.sampleLen_);
    */
   const startIndex =
        Math.round(Math.random() * ( this.textlist.length - 1));
  
    return [ this.textlist[startIndex], this.textToIndices(this.textlist[startIndex])];
  }

  /**
   * Get a slice of the training text data.
   *
   * @param {number} startIndex
   * @param {number} endIndex
   * @param {bool} useIndices Whether to return the indices instead of string.
   * @returns {string | Uint16Array} The result of the slicing.
   */
  slice_(startIndex, endIndex) {
    return this.textString_.slice(startIndex, endIndex);
  }

  /**
   * Get the set of unique characters from text.
   */
  getCharSet_() {
    this.charSet_ = [];
    for (let i = 0; i < this.textLen_; ++i) {
      if (this.charSet_.indexOf(this.textString_[i]) === -1) {
        this.charSet_.push(this.textString_[i]);      
      }
    }
    this.charSetSize_ = this.charSet_.length;
  }

  /**
   * Convert all training text to integer indices.
   */
  convertAllTextToIndices_() {
    /*
    var textExamples = ""
    for(let le = 0; le < this.textlist.length;le++)
    {
      textExamples = textExamples + this.textlist[le]
    }
    */
    this.indices_ = new Uint16Array(this.textToIndices(this.textString_));
  }

  /**
   * Generate the example-begin indices; shuffle them randomly.
   */
  generateExampleBeginIndices_() {
    // Prepare beginning indices of examples.
    this.exampleBeginIndices_ = [];
    for (let i = 0;
        i < this.textLen_ - this.sampleLen_ - 1;
        i += this.sampleStep_) {
      this.exampleBeginIndices_.push(i);
    }
    this.exampleBeginIndices_ = JSON.parse(JSON.stringify(this.textlist))
    // Randomly shuffle the beginning indices.
    tf.util.shuffle(this.textlist);
    this.examplePosition_ = 0;
  }
}

/**
 * Get a file by downloading it if necessary.
 *
 * @param {string} sourceURL URL to download the file from.
 * @param {string} destPath Destination file path on local filesystem.
 */
export async function maybeDownload(sourceURL, destPath) {
  const fs = require('fs');
  return new Promise(async (resolve, reject) => {
    if (!fs.existsSync(destPath) || fs.lstatSync(destPath).size === 0) {
      const localZipFile = fs.createWriteStream(destPath);
      console.log(`Downloading file from ${sourceURL} to ${destPath}...`);
      https.get(sourceURL, response => {
        response.pipe(localZipFile);
        localZipFile.on('finish', () => {
          localZipFile.close(() => resolve());
        });
        localZipFile.on('error', err => reject(err));
      });
    } else {
      return resolve();
    }
  });
}
